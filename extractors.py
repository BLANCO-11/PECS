import re
import json
from typing import List, Tuple, Dict, Optional
from groq import Groq

from config import AlphaConfig
class SymbolicExtractor:
    def process(self, text: str) -> Optional[List[Tuple]]:
        """
        Returns a list of (subject, predicate, object) tuples if matched.
        Returns None if ambiguous (requires LLM).
        """
        text = text.lower().strip()
        triples = []

        # Rule 1: Preference "I prefer X"
        match = re.search(r"i prefer ([\w\s]+)", text)
        if match:
            triples.append(("user", "prefers", match.group(1).strip()))
            return triples

        # Rule 2: Definition "X is a Y"
        match = re.search(r"([\w\s]+) is a ([\w\s]+)", text)
        if match:
            triples.append((match.group(1).strip(), "is_a", match.group(2).strip()))
            return triples
            
        # Rule 3: Comparison "X is better than Y"
        match = re.search(r"([\w\s]+) is better than ([\w\s]+)", text)
        if match:
            triples.append((match.group(1).strip(), "preferred_over", match.group(2).strip()))
            return triples

        return None

class LLMExtractor:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def extract(self, text: str, context_beliefs: List[Dict], verbose: bool = False, focus_topic: str = None) -> Dict:
        """
        Fallback extraction using Groq.
        """
        # Safety: Limit context to avoid token overflow
        context_str = "\n".join([f"- ({b['subject']}, {b['predicate']}, {b['object']})" for b in context_beliefs[:15]])
        
        if verbose:
            print(f"--- [DEBUG] Sending to LLM (Extract Context) ---\n{context_str}\n------------------------------------------------")
        
        focus_instruction = ""
        if focus_topic:
            focus_instruction = f"""
        IMPORTANT: The user is specifically interested in '{focus_topic}'. 
        - STRICTLY FILTER: Only extract facts that are directly relevant to '{focus_topic}'.
        - EXCLUSION: If the text contains information about other aspects (e.g. 'playing career' when focus is 'manager'), DO NOT extract them.
        - CONTEXT: Only include context if it explains '{focus_topic}'."""

        prompt = f"""
        Analyze the User Input and extract structured knowledge triples (Subject, Predicate, Object).
        
        Guidelines for Extraction:
        1. **Entity Resolution**: 
           - Resolve pronouns (he, she, it, they) and references (e.g., "the company") to their specific entities.
           - Use the provided 'Current Context' or the input text itself to find the correct reference.
        
        2. **Predicate Specificity**: 
           - Avoid generic verbs like "has", "is", "related_to", "involves". 
           - Use precise, descriptive verbs that capture the specific nature of the relationship (e.g., "authored", "commanded", "located_in", "diagnosed_with").
        
        3. **Atomicity**: 
           - Break complex sentences into multiple simple triples.
           - Ensure subjects and objects are standalone entities.
        
        {focus_instruction}
        Current Context:
        {context_str}
        
        User Input: "{text}"
        
        Return JSON ONLY. Schema:
        {{
          "proposed_beliefs": [
            {{"subject": "str", "predicate": "str", "object": "str", "confidence": float}}
          ]
        }}
        """
        
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "system", "content": "You are a JSON extractor."},
                          {"role": "user", "content": prompt}],
                model=AlphaConfig.EXTRACTION_MODEL, # Optimized: Lightweight model for structure
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            if verbose and completion.usage:
                u = completion.usage
                print(f"  [LLM Extract] Prompt: {u.prompt_tokens} | Completion: {u.completion_tokens} | Total: {u.total_tokens}")
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"LLM Error: {e}")
            return {"proposed_beliefs": []}

    def reason(self, text: str, active_beliefs: List[Dict], verbose: bool = False, model: str = AlphaConfig.REASONING_MODEL_FAST) -> str:
        """
        High-level reasoning when deterministic logic fails.
        """
        # Safety: Limit context to prevent token overflow, though core usually handles this.
        # We use the limit passed by the caller (Core), allowing for Deep Think modes.
        triples = [f"{b['subject']} {b['predicate']} {b['object']}" for b in active_beliefs]
        
        if verbose:
            print(f"--- [DEBUG] Sending to LLM (Reason Context) ---\n{triples}\n-----------------------------------------------")

        prompt = f"""
        You are an AI assistant using a retrieved memory bank.
        
        Memory Context:
        {json.dumps(triples, indent=2)}
        
        User Question: "{text}"
        
        Task:
        Answer the question comprehensively using ONLY the Memory Context. 
        Synthesize the facts into a coherent paragraph. 
        If the memory contains details about roles, achievements, or definitions, include them.
        Do not use external knowledge.
        """
        
        completion = self.client.chat.completions.create(
            messages=[{"role": "system", "content": "You are a helpful assistant grounded in provided facts."},
                      {"role": "user", "content": prompt}],
            model=model,
            temperature=0.1
        )
        if verbose and completion.usage:
            u = completion.usage
            print(f"  [LLM Reason]  Prompt: {u.prompt_tokens} | Completion: {u.completion_tokens} | Total: {u.total_tokens}")
        return completion.choices[0].message.content

    def synthesize(self, query: str, beliefs: List[Dict], verbose: bool = False) -> str:
        """
        Synthesizes a natural language response from a specific set of beliefs.
        Used for high-confidence/deterministic answers to sound natural.
        """
        triples = [f"{b['subject']} {b['predicate']} {b['object']}" for b in beliefs]
        # print(triples)
        # print("  [LLM Synth]  LOCAL FACTS")
        # print(*triples, sep="\n")
        
        prompt = f"""
        User Query: "{query}"
        
        Turn these facts into a natural response that answers the query. 
        Rules:
        1. Smooth out the phrasing to sound natural.
        2. Do NOT add new info or external knowledge.
        3. Stick strictly to the facts provided. Dont use a fact in the prompt if it is not relevant to query.
        
        Facts: {json.dumps(triples)}
        """
        
        completion = self.client.chat.completions.create(
            messages=[{"role": "system", "content": "You are a helpful synthesizer. You only rephrase provided facts."},
                      {"role": "user", "content": prompt}],
            model=AlphaConfig.SYNTHESIZE_MODEL,
            temperature=0.3
        )
        if verbose and completion.usage:
            u = completion.usage
            print(f"  [LLM Synth]   Prompt: {u.prompt_tokens} | Completion: {u.completion_tokens} | Total: {u.total_tokens}")
        return completion.choices[0].message.content

    def suggest_merges(self, beliefs: List[Dict]) -> List[Dict]:
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
            result = json.loads(completion.choices[0].message.content)
            
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

    def plan_research(self, topic: str) -> List[str]:
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
            result = json.loads(completion.choices[0].message.content)
            return result.get("sub_topics", [])
        except Exception as e:
            print(f"Research Planning Error: {e}")
            return []

    def check_relevance(self, root_topic: str, sub_topic: str) -> bool:
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
            response = completion.choices[0].message.content.strip().lower()
            return "yes" in response
        except Exception:
            return False

    def check_search_result_relevance(self, topic: str, title: str, snippet: str) -> bool:
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
            response = completion.choices[0].message.content.strip().lower()
            return "yes" in response
        except Exception:
            return True