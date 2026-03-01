import re
import json
from typing import List, Tuple, Dict, Optional
from groq import Groq

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

    def extract(self, text: str, context_beliefs: List[Dict], verbose: bool = False) -> Dict:
        """
        Fallback extraction using Groq.
        """
        context_str = "\n".join([f"- ({b['subject']}, {b['predicate']}, {b['object']})" for b in context_beliefs])
        
        if verbose:
            print(f"--- [DEBUG] Sending to LLM (Extract Context) ---\n{context_str}\n------------------------------------------------")
        
        prompt = f"""
        You are the Cognitive Core of an AI. Extract structured beliefs from the user input.
        
        Current Context:
        {context_str}
        
        User Input: "{text}"
        
        Return JSON ONLY. Schema:
        {{
          "proposed_beliefs": [
            {{"subject": "str", "predicate": "str", "object": "str", "type": "fact|preference"}}
          ]
        }}
        """
        
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "system", "content": "You are a JSON extractor."},
                          {"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant", # Optimized: Lightweight model for structure
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

    def reason(self, text: str, active_beliefs: List[Dict], verbose: bool = False) -> str:
        """
        High-level reasoning when deterministic logic fails.
        """
        triples = [f"({b['subject']} {b['predicate']} {b['object']})" for b in active_beliefs]
        
        if verbose:
            print(f"--- [DEBUG] Sending to LLM (Reason Context) ---\n{triples}\n-----------------------------------------------")

        prompt = f"""
        You are a bounded AI agent with NO external knowledge.
        You must answer the user's question based STRICTLY on the provided Context Beliefs.
        
        Context Beliefs: {triples}
        User Input: {text}
        
        Rules:
        1. If the answer cannot be derived from the Context Beliefs, say "I don't have enough information in my memory."
        2. Do NOT use any outside knowledge, facts, or assumptions.
        3. Do NOT hallucinate.
        """
        
        completion = self.client.chat.completions.create(
            messages=[{"role": "system", "content": "You are a bounded AI agent. You have no knowledge outside the provided context."},
                      {"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.0
        )
        if verbose and completion.usage:
            u = completion.usage
            print(f"  [LLM Reason]  Prompt: {u.prompt_tokens} | Completion: {u.completion_tokens} | Total: {u.total_tokens}")
        return completion.choices[0].message.content

    def synthesize(self, beliefs: List[Dict], verbose: bool = False) -> str:
        """
        Synthesizes a natural language response from a specific set of beliefs.
        Used for high-confidence/deterministic answers to sound natural.
        """
        triples = [f"{b['subject']} {b['predicate']} {b['object']}" for b in beliefs]
        # print(triples)
        prompt = f"""
        Turn these facts into a natural response. 
        Rules:
        1. Do NOT add new info or external knowledge.
        2. Stick strictly to the facts provided.
        
        Facts: {json.dumps(triples)}
        """
        
        completion = self.client.chat.completions.create(
            messages=[{"role": "system", "content": "You are a helpful synthesizer. You only rephrase provided facts."},
                      {"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
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
        
        Return JSON list of pairs to merge:
        [ {{"keep_id": "...", "merge_id": "..."}} ]
        """
        
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "system", "content": "You are a Knowledge Graph cleaner."},
                          {"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                response_format={"type": "json_object"}
            )
            result = json.loads(completion.choices[0].message.content)
            if isinstance(result, list):
                return result
            return result.get("pairs", [])
        except Exception as e:
            print(f"Merge Error: {e}")
            return []