import re
import time
import random
from collections import deque
from typing import List, Tuple
from storage import MemoryStore
from extractors import SymbolicExtractor, LLMExtractor
from config import AlphaConfig
from confidence import compute_confidence

class AlphaCognitiveCore:
    def __init__(self, groq_api_key: str, deep_think_mode: bool = False):
        self.memory = MemoryStore()
        self.symbolic = SymbolicExtractor()
        self.llm = LLMExtractor(groq_api_key)
        self.deep_think_mode = deep_think_mode
        
        # Resource Awareness & Curiosity
        self.llm_call_history = deque(maxlen=AlphaConfig.LLM_BUDGET_WINDOW)
        self.curiosity_goals_history = deque(maxlen=AlphaConfig.MAX_CURIOSITY_GOALS_WINDOW)
        self.current_interaction_llm_calls = 0
        self.session_new_beliefs_count = 0
        
    def learn(self, user_input: str, verbose: bool = False) -> List[Tuple]:
        """
        Ingests knowledge from input. Runs extraction and belief revision.
        Does NOT generate a response.
        """
        print(f"\n--- Learning from: '{user_input}' ---")
        
        # Always get context first to establish relationships for new beliefs
        raw_context = self._get_relevant_context(user_input)
        
        # Sort by Relevance (Keyword Overlap), then Direct Match, then Confidence
        raw_context.sort(key=lambda x: (x.get('relevance_score', 0), x.get('is_direct_match', False), x.get('confidence', 0)), reverse=True)
        context = raw_context[:AlphaConfig.TOP_K_ACTIVATION]

        # 1. Symbolic Extraction
        extracted_triples = self.symbolic.process(user_input)
        
        # 2. Fallback to LLM Extraction (if symbolic failed)
        if not extracted_triples:
            print("Symbolic extraction ambiguous. Calling LLM...")
            self.current_interaction_llm_calls += 1
            llm_result = self.llm.extract(user_input, context, verbose=verbose)
            
            extracted_triples = []
            for item in llm_result.get("proposed_beliefs", []):
                s = item.get('subject')
                p = item.get('predicate')
                o = item.get('object')
                if s and p and o:
                    extracted_triples.append((s, p, o))
            
            # Deduplicate to prevent LLM repetition loops
            extracted_triples = list(set(extracted_triples))
            
            if not extracted_triples:
                print("  [WARN] LLM returned no beliefs.")
        else:
            print(f"Symbolic match: {extracted_triples}")

        # 3. Belief Revision (Update Graph)
        newly_created_ids = []
        new_belief_content = {}
        for subj, pred, obj in extracted_triples:
            # Normalize
            subj, pred, obj = subj.lower(), pred.lower(), obj.lower()
            
            # Add/Update Belief
            b_id, is_new = self.memory.add_belief(subj, pred, obj)
            action = "Created" if is_new else "Strengthened"
            print(f"{action} belief: ({subj}, {pred}, {obj})")
            if is_new:
                newly_created_ids.append(b_id)
                new_belief_content[b_id] = (subj, obj)

        # Create edges for newly created beliefs, linking them to the context
        for new_id in newly_created_ids:
            n_subj, n_obj = new_belief_content[new_id]
            n_tokens = set(re.findall(r'\w+', f"{n_subj} {n_obj}".lower())) - AlphaConfig.STOP_WORDS
            
            for context_belief in context:
                if new_id != context_belief['id']:
                    # Semantic Overlap Check: Only link if they share meaningful nouns
                    c_tokens = set(re.findall(r'\w+', f"{context_belief['subject']} {context_belief['object']}".lower())) - AlphaConfig.STOP_WORDS
                    
                    if not n_tokens.isdisjoint(c_tokens):
                        self.memory.add_edge(source_id=context_belief['id'], target_id=new_id, edge_type='related_to', weight=0.5)
        
        self.memory.conn.commit()
        self.memory.prune_graph() # Enforce bounds
        
        # Resting Period: Check if we need to consolidate (Prune & Merge)
        self.session_new_beliefs_count += len(newly_created_ids)
        if self.session_new_beliefs_count >= AlphaConfig.CONSOLIDATION_THRESHOLD:
            print(f"\n[System] Resting period: Consolidating after {self.session_new_beliefs_count} new beliefs...")
            self.consolidate()
            self.session_new_beliefs_count = 0 # Reset counter

        return extracted_triples

    def batch_learn(self, text: str, verbose: bool = False) -> List[Tuple]:
        """
        Splits long text into chunks and learns from them sequentially.
        Efficient for ingesting news articles or documents.
        """
        print(f"\n--- Batch Learning ({len(text)} chars) ---")
        # Split by sentence-ending punctuation to respect semantics
        chunks = re.split(r'(?<=[.!?])\s+', text)
        
        current_batch = []
        current_len = 0
        all_triples = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk: continue
            
            # Group chunks into blocks of ~500 chars for efficient LLM extraction
            if current_len + len(chunk) > 500:
                triples = self.learn(" ".join(current_batch), verbose)
                all_triples.extend(triples)
                current_batch = [chunk]
                current_len = len(chunk)
            else:
                current_batch.append(chunk)
                current_len += len(chunk)
        
        # Process remaining
        if current_batch:
            triples = self.learn(" ".join(current_batch), verbose)
            all_triples.extend(triples)
            
        return all_triples

    def query(self, user_input: str, verbose: bool = False) -> str:
        """
        Retrieves context and generates a response. Does NOT learn.
        """
        print(f"\n--- Querying: '{user_input}' ---")
        
        # 4. Activation
        raw_context = self._get_relevant_context(user_input)
        
        # --- Autonomous Self-Correction ---
        # If we know nothing about the topic, research it immediately before answering.
        if not raw_context:
            # Extract potential topic (Proper Nouns)
            proper_nouns = re.findall(r'\b[A-Z][a-zA-Z0-9]+\b', user_input)
            keywords = [t for t in proper_nouns if t.lower() not in AlphaConfig.STOP_WORDS and len(t) > 3]
            
            if keywords:
                # Pick the most specific (longest) one
                keywords.sort(key=len, reverse=True)
                topic_to_research = keywords[0]
                print(f"[Self-Correction] I don't have information on '{topic_to_research}'. Researching now...")
                self.research_topic(topic_to_research, verbose=verbose)
                
                # Retry context fetch after learning
                raw_context = self._get_relevant_context(user_input)
        
        # Filter out low-confidence beliefs to reduce noise
        active_context = [b for b in raw_context if b['confidence'] >= AlphaConfig.MIN_CONFIDENCE_FOR_CONTEXT]
        
        # Sort by Relevance, then Direct Match, then Confidence
        active_context.sort(key=lambda x: (x.get('relevance_score', 0), x.get('is_direct_match', False), x['confidence']), reverse=True)
        limit = AlphaConfig.TOP_K_DEEP_THINK if self.deep_think_mode else AlphaConfig.TOP_K_ACTIVATION
        active_context = active_context[:limit]
        
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
            elif b['confidence'] > 0.75 and is_relevant:
                matches.append(b)

        if matches:
            # Optimization: Template-based response (Zero Tokens)
            # If we have a direct match and a known predicate, construct sentence locally.
            # Skip optimization if deep_think is requested to allow for richer synthesis.
            if not self.deep_think_mode:
                template_responses = []
                for b in matches[:AlphaConfig.TOP_K_SYNTHESIZE]:
                    pred_phrase = AlphaConfig.PREDICATE_MAP.get(b['predicate'])
                    if pred_phrase:
                        template_responses.append(f"{b['subject']} {pred_phrase} {b['object']}.")
                
                if template_responses:
                    return " ".join(template_responses)

            # Select up to 3 .llm.synthesize(selected, verbose=verbose)
            synth_limit = AlphaConfig.TOP_K_DEEP_THINK if self.deep_think_mode else AlphaConfig.TOP_K_SYNTHESIZE
            selected = matches[:synth_limit]
            # Use lightweight LLM to generate a natural sentence from the facts
            response = self.llm.synthesize(user_input, selected, verbose=verbose)


        # 6. LLM Reasoner (Fallback)
        if not response:
            # Optimization: If we know nothing relevant, don't hallucinate or waste tokens.
            if not active_context:
                return "I don't have enough information in my memory to answer that."

            print("Deterministic logic insufficient. Calling LLM Reasoner...")
            self.current_interaction_llm_calls += 1
            response = self.llm.reason(user_input, active_context, verbose=verbose)

        print(f"RESPONSE: {response}")
        return response

    def process_interaction(self, user_input: str, verbose: bool = False):
        """
        Conversational wrapper: Decides whether to learn, then responds.
        """
        self.current_interaction_llm_calls = 0 # Reset for this interaction

        # Command: Explicit Research
        # Allows user to say "Learn about football" or "Research quantum physics"
        if user_input.lower().startswith(("learn about ", "research ")):
            topic = re.sub(r"^(learn about|research)\s+", "", user_input, flags=re.IGNORECASE).strip()
            self.research_topic(topic, verbose)
            return f"I have finished researching '{topic}' and updated my memory."

        # Optimization: If input is a question, skip extraction to save tokens
        is_question = "?" in user_input or user_input.lower().strip().startswith(
            ("what", "who", "where", "when", "why", "how", "tell", "describe", "explain", "show")
        )
        
        if not is_question:
            self.learn(user_input, verbose)
        else:
            print(f"\n--- Processing: '{user_input}' (Read-Only) ---")
            
        response = self.query(user_input, verbose)

        # Record usage and check curiosity
        self.llm_call_history.append(self.current_interaction_llm_calls)
        self.evaluate_curiosity_trigger(verbose)

        return response

    def consolidate(self):
        """
        Maintenance task: Looks for redundant beliefs and merges them.
        """
        print("\n--- Running Consolidation Tick ---")
        
        # 1. Forget weak/stale beliefs first (Cleanup)
        self.memory.forget_weak_beliefs()

        # 2. Recompute Structural Support
        self.memory.recompute_structural_support()
        
        # 3. Fetch a batch of beliefs (e.g., top 50 most recently updated)
        # For simplicity, we fetch all (in a real system, paginate this)
        all_beliefs = self.memory.get_all_beliefs()
        if len(all_beliefs) < 2: return
        
        # 4. Ask LLM for merges
        self.current_interaction_llm_calls += 1 # Tracking maintenance calls too
        merges = self.llm.suggest_merges(all_beliefs[:15]) # Limit to 15 for token limits
        
        # 5. Apply merges
        for m in merges:
            print(f"Merging {m['merge_id']} -> {m['keep_id']}")
            self.memory.merge_beliefs(m['keep_id'], m['merge_id'])
            
        # 6. Check Curiosity
        self.evaluate_curiosity_trigger()

    def research_topic(self, topic: str, depth: int = 0, verbose: bool = False, root_topic: str = None, is_autonomous: bool = False, start_time: float = None):
        """
        Actively searches Wikipedia for a topic and learns from the summary.
        Recursively explores interesting sub-topics found during research.
        Respects a time budget to prevent non-stop processing.
        
        Args:
            root_topic: The original query (e.g. "Spain") to keep research focused.
            is_autonomous: If True, allows drifting by creating new goals for unrelated sparks.
            start_time: Timestamp of when the top-level research began.
        """
        if start_time is None:
            start_time = time.time()

        # Check Time Budget
        elapsed = time.time() - start_time
        if elapsed > AlphaConfig.RESEARCH_MAX_TIME_SECONDS:
            if verbose: print(f"[Research] ⏳ Time budget exceeded ({elapsed:.1f}s). Deferring '{topic}' to future goals.")
            # Save the current topic as a goal to resume later
            self.memory.add_goal(f"Research {topic}", priority=5)
            return

        if depth > 2: # Limit recursion to prevent infinite rabbit holes
            return
            
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
        learned_triples = self.batch_learn(snippet, verbose=verbose)
        
        # --- Structural Research (Autonomy) ---
        # If this is the main topic (depth 0), we plan a structured deep dive.
        if depth == 0:
            if not root_topic: root_topic = topic
            print(f"[Research] Realizing I need a structured understanding of '{topic}'...")
            sub_topics = self.llm.plan_research(topic)
            
            if sub_topics:
                print(f"[Research] Plan created: {sub_topics}")
                for sub in sub_topics:
                    # Utilize Goal Creation to track what we are going to learn
                    goal_desc = f"Research {topic}: {sub}"
                    g_id = self.memory.add_goal(goal_desc, priority=10) # High priority to do it now
                    
                    # Execute immediately to fulfill the user's request in this session
                    print(f"[Research] Executing sub-goal: {sub}")
                    self.research_topic(sub, depth=depth + 1, verbose=verbose, root_topic=root_topic, is_autonomous=is_autonomous, start_time=start_time)
                    
                    # Mark done
                    if g_id: self.memory.complete_goal(g_id)
                    
                return # Plan completed

        # --- Serendipitous Curiosity ---
        # If we are exploring a sub-topic or no plan was made, follow interesting sparks.
        sparks = self._identify_curiosity_sparks(learned_triples)
        for spark in list(sparks)[:2]: # Limit to top 2 sparks
            if is_autonomous:
                # Autonomous Mode: Fine to sway away.
                # Instead of diving NOW (which distracts from current topic), add as a FUTURE goal.
                print(f"[Research] Autonomous: Found interesting spark '{spark}'. Adding to future goals.")
                self.memory.add_goal(f"Research {spark}", priority=5)
            else:
                # User Mode: Stick to the original query.
                # Only dive if strictly relevant to the root topic.
                if root_topic and self.llm.check_relevance(root_topic, spark):
                    print(f"[Research] Focused: '{spark}' is relevant to '{root_topic}'. Diving deeper...")
                    self.research_topic(spark, depth=depth + 1, verbose=verbose, root_topic=root_topic, is_autonomous=False, start_time=start_time)
                else:
                    if verbose: print(f"[Research] Focused: Ignoring '{spark}' as it is not relevant to '{root_topic}'.")

    def autonomous_discovery(self, verbose: bool = False, max_cycles: int = None):
        """
        The 'Inner Voice' loop. Fetches news, learns, and decides if it needs to research more.
        """
        print("\n--- 🧠 Starting Autonomous Discovery Loop (Press Ctrl+C to stop) ---")
        try:
            import tools # Lazy import to avoid circular dependencies if any
        except ImportError:
            print("[System] Warning: 'tools.py' not found. External news scanning will be skipped.")
            tools = None
        import time
        
        seen_urls = set()
        cycles = 0
        
        try:
            while True:
                if max_cycles and cycles >= max_cycles:
                    print(f"[System] Reached max autonomous cycles ({max_cycles}). Stopping.")
                    break
                cycles += 1

                # 0. Check Curiosity Trigger (Internal Drive)
                self.evaluate_curiosity_trigger(verbose)

                # 1. Goal Execution (Priority)
                next_goal = self.memory.get_next_goal()
                if next_goal:
                    print(f"\n[Goal] Executing: {next_goal['description']}")
                    if next_goal['description'].startswith("Research "):
                        topic = next_goal['description'].replace("Research ", "")
                        self.research_topic(topic, verbose=verbose, is_autonomous=True)
                        self.memory.complete_goal(next_goal['id'])
                        print(f"[Goal] Marked '{next_goal['description']}' as achieved.")
                    
                    # Short pause between goals, then loop back to check for more goals
                    time.sleep(2) 
                    continue

                # 2. Fetch News (Idle behavior)
                if tools:
                    print("\n[Inner Voice] No pending goals. Scanning global news feeds...")
                    news_items = tools.fetch_rss_news()
                else:
                    print("\n[Inner Voice] No pending goals. Reflecting on internal knowledge...")
                    news_items = []
                
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
                    unknown_subjects = self._identify_curiosity_sparks(triples)
                    
                    if unknown_subjects:
                        # 4. Action: Research
                        print(f"\n[Inner Voice] I learned something new about {list(unknown_subjects)}. I feel curious to know more.")
                        # Note: We skip reading the full news article here to focus on the goal-based research
                        # which is cleaner. If you want to read the news article too, you'd need a robust scraper.
                        # For now, we rely on the headline for the "spark" and Wikipedia for the "deep dive".
                        
                        # Create a goal for deep dive instead of immediate action
                        focus_topic = list(unknown_subjects)[0]
                        print(f"[Inner Voice] Creating goal: Research '{focus_topic}'")
                        self.memory.add_goal(f"Research {focus_topic}", priority=5)
                    else:
                        print("[Inner Voice] I am already familiar with these concepts. Moving on.")
                
                if new_items_count == 0:
                    print("No new headlines found. Sleeping...")
                
                time.sleep(60)

        except KeyboardInterrupt:
            print("\n[System] Autonomous discovery stopped by user.")

    def _identify_curiosity_sparks(self, triples: List[Tuple]) -> set:
        """Identifies subjects in the provided triples that the system knows very little about."""
        unknown_subjects = set()
        for subj, pred, obj in triples:
            # Check if this subject is "new" to us (low connectivity)
            related = self.memory.search_beliefs([subj])
            # If we have <= 1 belief (the one we just added), we are curious
            if len(related) <= 1:
                unknown_subjects.add(subj)
        return unknown_subjects

    # --- Curiosity Engine ---

    def evaluate_curiosity_trigger(self, verbose: bool = False):
        # 1. Check Pending Goals
        if self.memory.get_next_goal():
            return
            
        # 2. Check LLM Budget
        total_calls = sum(self.llm_call_history)
        if total_calls > AlphaConfig.MAX_LLM_CALLS_IN_WINDOW:
            if verbose: print("[Curiosity] Suppressed: LLM budget exceeded.")
            return

        # 3. Check Cooldown
        now = time.time()
        if self.curiosity_goals_history:
            last_time = self.curiosity_goals_history[-1]
            if now - last_time < AlphaConfig.CURIOSITY_COOLDOWN_SECONDS:
                return

        # 4. Check Stability
        stability = self.compute_system_stability()
        if stability < AlphaConfig.STABILITY_THRESHOLD:
            if verbose: print(f"[Curiosity] Suppressed: System unstable ({stability:.2f}).")
            return
            
        # 5. Compute Curiosity & Select
        top_belief = self.get_most_curious_belief()
        if top_belief and top_belief.get('curiosity', 0) > AlphaConfig.CURIOSITY_THRESHOLD:
            self.create_curiosity_goal(top_belief, verbose)

    def compute_system_stability(self):
        # Average confidence of top 20 beliefs (by evidence)
        beliefs = self.memory.get_all_beliefs()
        if not beliefs: return 0.0
        
        # Sort by evidence score
        top_beliefs = sorted(beliefs, key=lambda x: x['evidence_score'], reverse=True)[:20]
        
        total_conf = sum(compute_confidence(b) for b in top_beliefs)
        return total_conf / len(top_beliefs)

    def get_most_curious_belief(self):
        beliefs = self.memory.get_all_beliefs()
        edges = self.memory.get_edges()
        
        # Build adjacency for density
        adj = {}
        contradictions = {}
        for e in edges:
            s, t = e['source_id'], e['target_id']
            adj[s] = adj.get(s, 0) + 1
            adj[t] = adj.get(t, 0) + 1
            if e['type'] == 'contradicts':
                contradictions[s] = contradictions.get(s, 0) + 1
                contradictions[t] = contradictions.get(t, 0) + 1
                
        candidates = []
        for b in beliefs:
            conf = compute_confidence(b)
            usage = b['usage_count']
            novelty = 1.0 / (1.0 + usage)
            
            total_edges = adj.get(b['id'], 0)
            c_edges = contradictions.get(b['id'], 0)
            c_density = (c_edges / total_edges) if total_edges > 0 else 0.0
            
            # Allow curiosity for novelty even if no contradictions (min factor 0.1)
            c_factor = max(c_density, 0.1)
            support = b['structural_support_score']
            
            score = (1.0 - conf) * novelty * c_factor * (1.0 / (1.0 + support))
            b['curiosity'] = score
            candidates.append(b)
            
        candidates.sort(key=lambda x: x['curiosity'], reverse=True)
        return candidates[0] if candidates else None

    def create_curiosity_goal(self, belief, verbose: bool = False):
        # Simple Type C: Novelty/Research
        topic = belief['subject']
        print(f"\n[Curiosity] Triggered! Interested in '{topic}' (Score: {belief['curiosity']:.2f})")
        self.memory.add_goal(f"Research {topic}", priority=3)
        self.curiosity_goals_history.append(time.time())

    def _get_relevant_context(self, text: str) -> List[dict]:
        """Retrieves relevant beliefs using keyword search + 1-hop expansion."""
        # Optimization: Prefer proper nouns (capitalized words) to reduce noise.
        # This prevents common/short words like "la" (La Liga) from matching "Hillary" or "Plane".
        proper_nouns = re.findall(r'\b[A-Z][a-zA-Z0-9]+\b', text)
        keywords = [t.lower() for t in proper_nouns if t.lower() not in AlphaConfig.STOP_WORDS and len(t) > 3]
        
        # Fallback: If no proper nouns found, use standard tokens but keep length filter
        if not keywords:
            tokens = re.findall(r'\w+', text.lower())
            keywords = [t for t in tokens if t not in AlphaConfig.STOP_WORDS and len(t) > 3]
            
        if not keywords: return []
        
        # Sort by length (descending) to prioritize specific terms (e.g. "Cardiomyopathy" > "Fatal")
        keywords.sort(key=len, reverse=True)
        
        # Deduplicate preserving order (set() destroys order, potentially losing specific keywords)
        seen = set()
        unique_keywords = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique_keywords.append(k)
        
        seeds = self.memory.search_beliefs(unique_keywords[:10])
        if not seeds: return []
        
        seed_ids = [b['id'] for b in seeds]
        nodes, _ = self.memory.get_subgraph(seed_ids)
        
        seed_id_set = set(seed_ids)
        
        for node in nodes:
            node['confidence'] = compute_confidence(node)
            node['is_direct_match'] = node['id'] in seed_id_set
            
            # Relevance Scoring: Weight matches by keyword length (Specificity)
            # A belief matching "Sovietization" (len 13) gets higher score than "India" (len 5)
            score = 0
            matches = 0
            matched_tokens = set()
            content = f"{node['subject']} {node['predicate']} {node['object']}".lower()
            for k in unique_keywords:
                if k in content:
                    score += len(k)
                    matches += 1
                    matched_tokens.add(k)
            
            # Filter Noise: If we only match 1 keyword and it's a generic entity type (e.g. "League"), discard.
            if matches == 1 and list(matched_tokens)[0] in AlphaConfig.COMMON_ENTITIES:
                continue
                
            node['relevance_score'] = score
            
        return nodes

    def close(self):
        self.memory.close()