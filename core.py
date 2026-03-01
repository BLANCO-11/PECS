import re
from typing import List, Tuple
from storage import MemoryStore
from extractors import SymbolicExtractor, LLMExtractor
from activation import ActivationEngine
from config import AlphaConfig

class AlphaCognitiveCore:
    def __init__(self, groq_api_key: str):
        self.memory = MemoryStore()
        self.symbolic = SymbolicExtractor()
        self.llm = LLMExtractor(groq_api_key)
        self.activator = ActivationEngine(self.memory)
        
    def learn(self, user_input: str, verbose: bool = False) -> List[Tuple]:
        """
        Ingests knowledge from input. Runs extraction and belief revision.
        Does NOT generate a response.
        """
        print(f"\n--- Learning from: '{user_input}' ---")
        
        # Always get context first to establish relationships for new beliefs
        context = self.activator.get_activated_beliefs(user_input.lower().split())
        
        # 1. Symbolic Extraction
        extracted_triples = self.symbolic.process(user_input)
        
        # 2. Fallback to LLM Extraction (if symbolic failed)
        if not extracted_triples:
            print("Symbolic extraction ambiguous. Calling LLM...")
            llm_result = self.llm.extract(user_input, context, verbose=verbose)
            
            extracted_triples = []
            for item in llm_result.get("proposed_beliefs", []):
                s = item.get('subject')
                p = item.get('predicate')
                o = item.get('object')
                if s and p and o:
                    extracted_triples.append((s, p, o))
            if not extracted_triples:
                print("  [WARN] LLM returned no beliefs.")
        else:
            print(f"Symbolic match: {extracted_triples}")

        # 3. Belief Revision (Update Graph)
        newly_created_ids = []
        for subj, pred, obj in extracted_triples:
            # Normalize
            subj, pred, obj = subj.lower(), pred.lower(), obj.lower()
            
            # Add/Update Belief
            b_id, is_new = self.memory.add_belief(subj, pred, obj)
            action = "Created" if is_new else "Strengthened"
            print(f"{action} belief: ({subj}, {pred}, {obj})")
            if is_new:
                newly_created_ids.append(b_id)

        # Create edges for newly created beliefs, linking them to the context
        for new_id in newly_created_ids:
            for context_belief in context:
                if new_id != context_belief['id']:
                    self.memory.add_edge(source_id=context_belief['id'], target_id=new_id, edge_type='derived_from', weight=0.5)
        
        self.memory.conn.commit()
        self.memory.prune_graph() # Enforce bounds

        return extracted_triples

    def batch_learn(self, text: str, verbose: bool = False):
        """
        Splits long text into chunks and learns from them sequentially.
        Efficient for ingesting news articles or documents.
        """
        print(f"\n--- Batch Learning ({len(text)} chars) ---")
        # Split by sentence-ending punctuation to respect semantics
        chunks = re.split(r'(?<=[.!?])\s+', text)
        
        current_batch = []
        current_len = 0
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk: continue
            
            # Group chunks into blocks of ~500 chars for efficient LLM extraction
            if current_len + len(chunk) > 500:
                self.learn(" ".join(current_batch), verbose)
                current_batch = [chunk]
                current_len = len(chunk)
            else:
                current_batch.append(chunk)
                current_len += len(chunk)
        
        # Process remaining
        if current_batch:
            self.learn(" ".join(current_batch), verbose)

    def query(self, user_input: str, verbose: bool = False) -> str:
        """
        Retrieves context and generates a response. Does NOT learn.
        """
        print(f"\n--- Querying: '{user_input}' ---")
        
        # 4. Activation
        tokens = user_input.lower().split()
        raw_context = self.activator.get_activated_beliefs(tokens)
        
        # Filter out low-confidence beliefs to reduce noise
        active_context = [b for b in raw_context if b['confidence'] >= AlphaConfig.MIN_CONFIDENCE_FOR_CONTEXT]
        
        print(f"Activated {len(active_context)} beliefs (filtered from {len(raw_context)}).")
        if verbose:
            print("--- [DEBUG] Retrieved Context from DB ---")
            for b in active_context:
                print(f"  * {b['subject']} {b['predicate']} {b['object']} (Conf: {b['confidence']:.2f})")
            print("-----------------------------------------")

        # 5. Planning / Response
        # Simple deterministic check: Do we have a direct answer?
        response = None

        # Deterministic Matcher
        matches = []
        causal_terms = ["due to", "because", "caused", "reason", "fueling", "shaping", "linked", "tied", "driven", "result", "leads to"]

        # Prepare clean tokens for relevance check to filter out incidental matches
        stop_words = AlphaConfig.STOP_WORDS
        input_words = re.findall(r'\w+', user_input.lower())
        clean_input_tokens = {t for t in input_words if t not in stop_words}

        for b in active_context:
            is_causal = any(term in b['predicate'] for term in causal_terms)
            
            # Check if subject or object overlaps with meaningful input tokens
            b_content = set(re.findall(r'\w+', (b['subject'] or "").lower())) | set(re.findall(r'\w+', (b['object'] or "").lower()))
            is_relevant = not clean_input_tokens.isdisjoint(b_content)
            
            if "why" in user_input.lower():
                if is_causal and is_relevant:
                    matches.append(b)
            elif b['confidence'] > 0.85 and is_relevant:
                matches.append(b)

        if matches:
            # Select up to 3 matches to provide a richer, less static answer
            selected = matches[:3]
            # Use lightweight LLM to generate a natural sentence from the facts
            response = self.llm.synthesize(selected, verbose=verbose)

        # 6. LLM Reasoner (Fallback)
        if not response:
            # Optimization: If we know nothing relevant, don't hallucinate or waste tokens.
            if not active_context:
                return "I don't have enough information in my memory to answer that."

            print("Deterministic logic insufficient. Calling LLM Reasoner...")
            response = self.llm.reason(user_input, active_context, verbose=verbose)

        print(f"RESPONSE: {response}")
        return response

    def process_interaction(self, user_input: str, verbose: bool = False):
        """
        Conversational wrapper: Decides whether to learn, then responds.
        """
        # Optimization: If input is a question, skip extraction to save tokens
        is_question = "?" in user_input or user_input.lower().strip().startswith(("what", "who", "where", "when", "why", "how"))
        
        if not is_question:
            self.learn(user_input, verbose)
        else:
            print(f"\n--- Processing: '{user_input}' (Read-Only) ---")
            
        return self.query(user_input, verbose)

    def consolidate(self):
        """
        Maintenance task: Looks for redundant beliefs and merges them.
        """
        print("\n--- Running Consolidation Tick ---")
        
        # 1. Forget weak/stale beliefs first (Cleanup)
        self.memory.forget_weak_beliefs()
        
        # 2. Fetch a batch of beliefs (e.g., top 50 most recently updated)
        # For simplicity, we fetch all (in a real system, paginate this)
        all_beliefs = self.memory.get_all_beliefs()
        if len(all_beliefs) < 2: return
        
        # 3. Ask LLM for merges
        merges = self.llm.suggest_merges(all_beliefs[:50]) # Limit to 50 for token limits
        
        # 4. Apply merges
        for m in merges:
            print(f"Merging {m['merge_id']} -> {m['keep_id']}")
            self.memory.merge_beliefs(m['keep_id'], m['merge_id'])

    def research_topic(self, topic: str, verbose: bool = False):
        """
        Actively searches Wikipedia for a topic and learns from the summary.
        """
        print(f"\n[Research] Searching Wikipedia for: '{topic}'...")
        import tools
        
        url, summary = tools.search_wikipedia(topic)
        
        if not url or not summary:
            print("[Research] No Wikipedia results found.")
            return

        print(f"[Research] Found: {url}")
        print(f"[Research] Reading summary ({len(summary)} chars)...")
        
        # Limit extraction to 2000 chars (Wiki summaries are usually concise, but just in case)
        snippet = summary[:2000] 
        self.batch_learn(snippet, verbose=verbose)

    def autonomous_discovery(self, verbose: bool = False):
        """
        The 'Inner Voice' loop. Fetches news, learns, and decides if it needs to research more.
        """
        print("\n--- 🧠 Starting Autonomous Discovery Loop (Press Ctrl+C to stop) ---")
        import tools # Lazy import to avoid circular dependencies if any
        import time
        
        seen_urls = set()
        
        try:
            while True:
                # 1. Goal Execution (Priority)
                next_goal = self.memory.get_next_goal()
                if next_goal:
                    print(f"\n[Goal] Executing: {next_goal['description']}")
                    if next_goal['description'].startswith("Research "):
                        topic = next_goal['description'].replace("Research ", "")
                        self.research_topic(topic, verbose=verbose)
                        self.memory.complete_goal(next_goal['id'])
                        print(f"[Goal] Marked '{next_goal['description']}' as achieved.")
                    
                    # Short pause between goals, then loop back to check for more goals
                    time.sleep(2) 
                    continue

                # 2. Fetch News (Idle behavior)
                print("\n[Inner Voice] No pending goals. Scanning global news feeds...")
                news_items = tools.fetch_rss_news()
                
                if not news_items:
                    print("No news found or connection failed.")
                    news_items = []

                new_items_count = 0
                for item in news_items:
                    headline = item['title']
                    url = item['link']
                    
                    if url in seen_urls:
                        continue
                    
                    seen_urls.add(url)
                    new_items_count += 1
                    
                    print(f"\n[Attention] Found headline: '{headline}'")
                    
                    # 2. Initial Learn (Surface Level)
                    triples = self.learn(headline, verbose=False)
                    
                    # 3. Curiosity Check (The Inner Voice)
                    unknown_subjects = []
                    for subj, pred, obj in triples:
                        # Check if this subject is "new" to us (low connectivity)
                        related = self.memory.search_beliefs([subj])
                        # If we have <= 1 belief (the one we just added), we are curious
                        if len(related) <= 1:
                            unknown_subjects.append(subj)
                    
                    if unknown_subjects:
                        # 4. Action: Research
                        print(f"\n[Inner Voice] I learned something new about {unknown_subjects}. I feel curious to know more.")
                        # Note: We skip reading the full news article here to focus on the goal-based research
                        # which is cleaner. If you want to read the news article too, you'd need a robust scraper.
                        # For now, we rely on the headline for the "spark" and Wikipedia for the "deep dive".
                        
                        # Create a goal for deep dive instead of immediate action
                        focus_topic = unknown_subjects[0]
                        print(f"[Inner Voice] Creating goal: Research '{focus_topic}'")
                        self.memory.add_goal(f"Research {focus_topic}", priority=5)
                    else:
                        print("[Inner Voice] I am already familiar with these concepts. Moving on.")
                
                if new_items_count == 0:
                    print("No new headlines found. Sleeping...")
                
                time.sleep(60)

        except KeyboardInterrupt:
            print("\n[System] Autonomous discovery stopped by user.")

    def close(self):
        self.memory.close()