import re
import json
import spacy
from typing import Any, List, Tuple, Dict, Optional
from groq import Groq
from utils.config import AlphaConfig

class SpacyExtractor:
    """
    Optimized linguistic parser. 
    Performance: ~10ms per sentence on CPU.
    """
    def __init__(self):
        self.nlp = None
        try:
            # Check if model exists
            if not spacy.util.is_package("en_core_web_sm"):
                print("[System] Downloading optimized Spacy model...")
                from spacy.cli import download
                download("en_core_web_sm")
            
            # LOAD WITH OPTIMIZATIONS
            # We disable 'ner' (Named Entity Recognition) because it is computationally expensive
            # and we are strictly doing grammatical dependency parsing here.
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
            print("[System] Spacy loaded (CPU Optimized: NER disabled)")
        except Exception as e:
            print(f"[System] Warning: Spacy load failed ({e}). Falling back to Regex.")

    def process(self, text: str) -> List[Tuple]:
        if not self.nlp: return [] # Safety fallback
        
        doc = self.nlp(text)
        triples = []

        for token in doc:
            # Look for subjects (nominal subjects)
            if token.dep_ in ("nsubj", "nsubjpass"):
                subj = self._get_compound(token)
                verb = token.head.lemma_ # The predicate (lemma form)
                
                # Look for objects attached to the verb
                for child in token.head.children:
                    # Direct objects, attributes (is a ...), or complements
                    if child.dep_ in ("dobj", "attr", "acomp"):
                        obj = self._get_compound(child)
                        triples.append((subj, verb, obj))
                    
                    # Handle prepositional objects (e.g. "lived in London")
                    elif child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                obj = self._get_compound(grandchild)
                                full_verb = f"{verb}_{child.text}"
                                triples.append((subj, full_verb, obj))
        return triples

    def _get_compound(self, token):
        """Helper to capture multi-word entities (e.g. 'United States')."""
        compound = [child.text for child in token.children if child.dep_ == "compound"]
        compound.append(token.text)
        # Check for adjectives (e.g. 'Red Planet')
        for child in token.children:
            if child.dep_ == "amod":
                compound.insert(0, child.text)
        return " ".join(compound)

class SymbolicExtractor:
    def process(self, text: str) -> Optional[List[Tuple]]:
        """
        Ultra-fast Regex Fallback (0ms latency).
        Used if Spacy misses or fails.
        """
        text = text.lower().strip()
        triples = []

        def is_clean(s, o):
            if s in AlphaConfig.STOP_WORDS: return False
            if len(o) > 80 or '\n' in o: return False 
            return True

        # Rule 1: Preference "I prefer X"
        match = re.search(r"i prefer ([\w\s]+)", text)
        if match:
            s, o = "user", match.group(1).strip()
            if is_clean(s, o):
                triples.append((s, "prefers", o))

        # Rule 2: Definition "X is a Y"
        match = re.search(r"([\w\s]+) is a ([^\n]+)", text)
        if match:
            s, o = match.group(1).strip(), match.group(2).strip()
            if is_clean(s, o):
                triples.append((s, "is_a", o))
            
        # Rule 3: Comparison "X is better than Y"
        match = re.search(r"([\w\s]+) is better than ([^\n]+)", text)
        if match:
            s, o = match.group(1).strip(), match.group(2).strip()
            if is_clean(s, o):
                triples.append((s, "preferred_over", o))

        return triples if triples else None

class LLMExtractor:
    def __init__(self, api_key, token_callback=None):
        self.client = Groq(api_key=api_key)
        self.token_callback = token_callback

    def _parse_json_response(self, content: str) -> Any:
        """Helper to robustly parse JSON from LLM output, handling markdown blocks."""
        try:
            # Strip markdown code blocks if present
            content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'^```\s*', '', content, flags=re.MULTILINE)
            # Attempt to fix trailing commas which LLMs often output
            content = re.sub(r',\s*([\]}])', r'\1', content)
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return None

    def _record_usage(self, completion, label="LLM"):

        if hasattr(completion, "usage") and completion.usage:

            u = completion.usage

            if AlphaConfig.DEBUG_TOKENS:
                print(
                    f"[{label}] "
                    f"Prompt:{u.prompt_tokens} "
                    f"Completion:{u.completion_tokens} "
                    f"Total:{u.total_tokens}"
                )

            if self.token_callback:
                self.token_callback(u.total_tokens)

    def extract(self, text: str, context_beliefs: List[Dict], verbose: bool = False, focus_topic: str = None) -> Dict:
        # Safety: Limit context to avoid token overflow
        context_str = "\n".join([f"- ({b['subject']}, {b['predicate']}, {b['object']})" for b in context_beliefs[:15]])
        
        if verbose:
            print(f"--- [DEBUG] Sending to LLM (Extract Context) ---\n{context_str}\n------------------------------------------------")
        
        focus_instruction = ""
        if focus_topic:
            focus_instruction = f"""
        IMPORTANT: The user is specifically interested in '{focus_topic}'. 
        - STRICTLY FILTER: Only extract facts that are directly relevant to '{focus_topic}'.
        """

        prompt = f"""
        Analyze the User Input and extract structured knowledge triples (Subject, Predicate, Object).
        Guidelines: Resolve pronouns. Avoid generic verbs. Break complex sentences.
        {focus_instruction}
        Current Context:
        {context_str}
        User Input: "{text}"
        Return JSON ONLY. Schema: {{ "proposed_beliefs": [ {{"subject": "str", "predicate": "str", "object": "str", "confidence": float}} ] }}
        """
        
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "system", "content": "You are a JSON extractor."},
                          {"role": "user", "content": prompt}],
                model=AlphaConfig.EXTRACTION_MODEL,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            self._record_usage(completion)
            result = self._parse_json_response(completion.choices[0].message.content)
            return result if result else {"proposed_beliefs": []}
        except Exception as e:
            print(f"LLM Error: {e}")
            return {"proposed_beliefs": []}

    def reason(self, text: str, active_beliefs: List[Dict], verbose: bool=False,
           model: str=AlphaConfig.REASONING_MODEL_FAST) -> Dict:

        if not active_beliefs:
            return {"answer": None, "reasoning": None}

        # Limit context
        beliefs = active_beliefs[:AlphaConfig.MAX_REASON_FACTS]

        indexed_facts = []
        for i, b in enumerate(beliefs, 1):
            indexed_facts.append(f"[{i}] {b['subject']} {b['predicate']} {b['object']}")

        context_str = "\n".join(indexed_facts)

        prompt = f"""
        You are an AI assistant using a retrieved memory bank.

        Memory Context:
        {context_str}

        User Question: "{text}"

        Task:
        Answer the question using ONLY the Memory Context.

        Rules:
        - Only use the provided facts
        - Do NOT use outside knowledge
        - Every claim must reference fact IDs

        Format your response exactly as follows:

        RELEVANT_FACTS:
        [List the IDs (e.g. [1], [4]) of the facts that are relevant to the question.]

        REASONING:
        [Step-by-step reasoning using the fact IDs.]

        ANSWER:
        [Write a rich, natural explanation using the reasoning above. Have a natural response, dont include the reference to reasoning, only the response.]
        """

        completion = self.client.chat.completions.create(
            messages=[
                {"role":"system","content":"You are a grounded reasoning engine."},
                {"role":"user","content":prompt}
            ],
            model=model,
            temperature=0
        )

        self._record_usage(completion,"Reason")

        raw_response = completion.choices[0].message.content

        answer_match = re.search(r"ANSWER:\s*(.*)", raw_response, re.DOTALL)
        reasoning_match = re.search(r"REASONING:\s*(.*?)\nANSWER:", raw_response, re.DOTALL)

        answer = answer_match.group(1).strip() if answer_match else raw_response
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None

        return {
            "answer": answer,
            "reasoning": reasoning
        }
    
    def synthesize(self, query: str, beliefs: List[Dict], verbose: bool = False) -> str:

        if verbose:
            print(f"--- [DEBUG] Sending to LLM (Synthesize) ---\n")
            for b in beliefs:
                print(f"- ({b['subject']}, {b['predicate']}, {b['object']})")
            print("------------------------------------------------")
            
        
        triples = [f"{b['subject']} {b['predicate']} {b['object']}" for b in beliefs]
        prompt = f"""
        Query: "{query}"
        Facts: {json.dumps(triples)}
        Task: Answer the query naturally using only the facts. Do not add External Knowledge. The facts are your only source of Information.
        """
        completion = self.client.chat.completions.create(
            messages=[{"role": "system", "content": "Synthesizer."}, {"role": "user", "content": prompt}],
            model=AlphaConfig.SYNTHESIZE_MODEL,
            temperature=0.3
        )
        self._record_usage(completion, "Synthesize")
        return completion.choices[0].message.content

    def suggest_merges(self, beliefs: List[Dict], verbose: bool = False) -> List[Dict]:
        """
        Identifies beliefs that are semantically identical and should be merged.
        """
        # Simplify for LLM
        simple_list = [{"id": b['id'], "text": f"{b['subject']} {b['predicate']} {b['object']}"} for b in beliefs]
        
        prompt = f"""
        Analyze these beliefs. Identify pairs that mean the EXACT same thing but are phrased differently.
        Example: "US economy grew" and "United States GDP increased" -> Merge.
        
        Beliefs:
        {json.dumps(simple_list, indent=2)}
        
        Return a JSON object with a "merges" key containing a list of pairs:
        {{
            "merges": [ {{"keep_id": "...", "merge_id": "..."}} ]
        }}
        """
        
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "system", "content": "You are a Knowledge Graph cleaner."},
                          {"role": "user", "content": prompt}],
                model=AlphaConfig.MERGE_MODEL,
                response_format={"type": "json_object"}
            )
            self._record_usage(completion)
            if verbose and completion.usage:
                u = completion.usage
                print(f"  [LLM Merge]   Prompt: {u.prompt_tokens} | Completion: {u.completion_tokens} | Total: {u.total_tokens}")
            result = self._parse_json_response(completion.choices[0].message.content) or {}
            
            candidates = []
            if isinstance(result, list):
                candidates = result
            elif isinstance(result, dict):
                candidates = result.get("merges", result.get("pairs", []))

            # Validate keys to prevent KeyErrors downstream
            return [m for m in candidates if isinstance(m, dict) and "keep_id" in m and "merge_id" in m]
        except Exception as e:
            print(f"Merge Error: {e}")
            return []

    def plan_research(self, topic: str, verbose: bool = False) -> List[str]:
        """
        Generates a structured research plan (sub-topics) for a given topic.
        """
        prompt = f"""
        I need to research '{topic}' thoroughly.
        Identify 3 to 5 distinct sub-topics or related entities that are essential to understanding '{topic}'.
        
        CRITICAL GUIDELINES:
        1. **Relevance**: Sub-topics must be DIRECTLY related to '{topic}'. Avoid tangential, speculative, or minor associations.
        2. **Significance**: Focus on major entities, key events, or core concepts.
        3. **Searchability**: Use simple, standard names likely to have their own encyclopedia page.
        4. **Context**: If the topic specifies a role (e.g., "as a manager"), ONLY include sub-topics relevant to that specific role.
        
        Do NOT use sentences, complex phrases, or "Impact of..." titles. Do NOT hallucinate connections.
        
        Example:
        Topic: "Zinedine Zidane as a manager"
        Sub-topics: ["Real Madrid", "UEFA Champions League", "Florentino Pérez", "La Liga"]
        
        Return JSON ONLY in this format:
        {{
            "sub_topics": ["Entity 1", "Entity 2", ...]
        }}
        """
        
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "system", "content": "You are a research planner. You output Wikipedia-friendly search terms."},
                          {"role": "user", "content": prompt}],
                model=AlphaConfig.RESEARCH_PLANNER_MODEL,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            self._record_usage(completion)
            if verbose and completion.usage:
                u = completion.usage
                print(f"  [LLM Plan]    Prompt: {u.prompt_tokens} | Completion: {u.completion_tokens} | Total: {u.total_tokens}")
            result = self._parse_json_response(completion.choices[0].message.content) or {}
            return result.get("sub_topics", [])
        except Exception as e:
            print(f"Research Planning Error: {e}")
            return []

    def check_relevance(self, root_topic: str, sub_topic: str, verbose: bool = False) -> bool:
        """
        Determines if a sub_topic is strictly relevant to the root_topic.
        Used to prevent 'rabbit hole' drifting during focused research.
        """
        prompt = f"""
        I am researching the main topic: "{root_topic}"
        
        I have discovered a potential sub-topic: "{sub_topic}"
        
        Is this sub-topic directly related and useful for gaining a deeper understanding of the main topic?
        
        - Answer YES if it is a key concept, related field, important person/event, or foundational work.
        - Answer NO if it is a tangent, a minor detail, or an unrelated concept.
        
        Answer with only YES or NO.
        """
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=AlphaConfig.RELEVANCE_CHECK_MODEL,
                max_tokens=5
            )
            self._record_usage(completion)
            if verbose and completion.usage:
                u = completion.usage
                print(f"  [LLM Relev]   Prompt: {u.prompt_tokens} | Completion: {u.completion_tokens} | Total: {u.total_tokens}")
            response = completion.choices[0].message.content.strip().lower()
            return "yes" in response
        except Exception:
            return False

    def check_search_result_relevance(self, topic: str, title: str, snippet: str, verbose: bool = False) -> bool:
        """
        Determines if a search result is relevant to the research topic.
        """
        prompt = f"""
        I am researching: "{topic}"
        
        Search Result:
        Title: "{title}"
        Snippet: "{snippet}"
        
        Is this result relevant to the topic? Answer YES or NO.
        """
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=AlphaConfig.RELEVANCE_CHECK_MODEL,
                max_tokens=5
            )
            self._record_usage(completion)
            if verbose and completion.usage:
                u = completion.usage
                print(f"  [LLM Search]  Prompt: {u.prompt_tokens} | Completion: {u.completion_tokens} | Total: {u.total_tokens}")
            response = completion.choices[0].message.content.strip().lower()
            return "yes" in response
        except Exception:
            return True
        
    def check_contradictions_batch(self, candidate_pairs: List[Dict], verbose: bool = False) -> List[Dict]:
        if not candidate_pairs: return []
        payload = [{"id": i, "fact1": f"{p['fact1']}", "fact2": f"{p['fact2']}"} for i, p in enumerate(candidate_pairs)]
        
        prompt = f"""
        Determine logical contradictions.
        Pairs: {json.dumps(payload)}
        Return JSON: {{ "contradicting_ids": [int] }}
        """
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=AlphaConfig.EXTRACTION_MODEL,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            self._record_usage(completion)
            result = self._parse_json_response(completion.choices[0].message.content) or {}
            ids = result.get("contradicting_ids", [])
            return [candidate_pairs[i] for i in ids if i < len(candidate_pairs)]
        except Exception:
            return []
        
        
    def select_relevant_facts(self, query, beliefs, verbose=False):

        beliefs = beliefs[:AlphaConfig.FACT_SELECTOR_LIMIT]

        indexed = []

        for i,b in enumerate(beliefs,1):
            indexed.append(f"[{i}] {b['subject']} {b['predicate']} {b['object']}")

        context = "\n".join(indexed)

        prompt = f"""
                    Memory Context:
                    {context}

                    User Question:
                    "{query}"
                    
                    - Only use information explicitly present in the Memory Context.
                    - If something is not present in the facts, you must NOT mention it.
                    - Do not rely on outside knowledge.
                    - Every claim must correspond to one of the fact IDs.
                    - Select the facts that could help answer the question.
                    - Return only relevant facts to the query.
                    - Avoid selecting very technical details unless the question specifically asks for them.

                    Return JSON:
                    {{"facts":[1,3,5]}}
                """

        completion = self.client.chat.completions.create(
            messages=[{"role":"user","content":prompt}],
            model=AlphaConfig.REASONING_MODEL_FAST,
            temperature=0,
            response_format={"type":"json_object"}
        )

        self._record_usage(completion,"FactSelect")

        result = self._parse_json_response(completion.choices[0].message.content) or {}

        return result.get("facts",[])
