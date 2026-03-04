import re
import time
import random
from collections import deque
from typing import List, Tuple
from storage import MemoryStore
from extractors import SymbolicExtractor, LLMExtractor
from config import AlphaConfig
from confidence import compute_confidence
from activation import ActivationEngine

class PECSCore:
    def __init__(self, groq_api_key: str, deep_think_mode: bool = False):
        self.memory = MemoryStore()

        # --- Data Integrity Check ---
        # Clean up any orphaned edges from previous versions or unclean shutdowns
        # to prevent context retrieval errors like "None cannot be a node".
        if hasattr(self.memory, 'clean_orphaned_edges'):
            self.memory.clean_orphaned_edges()

        self.symbolic = SymbolicExtractor()
        self.llm = LLMExtractor(groq_api_key)
        self.deep_think_mode = deep_think_mode
        self.activation = ActivationEngine(self.memory)
        
        # Resource Awareness & Curiosity
        self.llm_call_history = deque(maxlen=AlphaConfig.LLM_BUDGET_WINDOW)
        self.curiosity_goals_history = deque(maxlen=AlphaConfig.MAX_CURIOSITY_GOALS_WINDOW)
        self.current_interaction_llm_calls = 0
        self.session_new_beliefs_count = 0
        
    def learn(self, user_input: str, verbose: bool = False, focus_topic: str = None, prune: bool = True) -> List[Tuple]:
        """
        Ingests knowledge from input. Runs extraction and belief revision.
        Does NOT generate a response.
        """
        # Clean and truncate input for logging
        clean_input = ' '.join(user_input.split())
        truncate_at = 70
        truncated_input = (clean_input[:truncate_at] + '...') if len(clean_input) > truncate_at else clean_input
        print(f"\n--- Learning from: '{truncated_input}' ---")
        
        # Always get context first to establish relationships for new beliefs
        raw_context = self._get_relevant_context(user_input)
        # Activation Engine already returns sorted, high-relevance beliefs
        context = raw_context

        # 1. Symbolic Extraction
        extracted_triples = self.symbolic.process(user_input)
        
        # 2. Fallback to LLM Extraction (if symbolic failed)
        if not extracted_triples:
            print("Symbolic extraction ambiguous. Calling LLM...")
            self.current_interaction_llm_calls += 1
            llm_result = self.llm.extract(user_input, context, verbose=verbose, focus_topic=focus_topic)
            
            extracted_triples = []
            for item in llm_result.get("proposed_beliefs", []):
                s = item.get('subject')
                p = item.get('predicate')
                o = item.get('object')
                confidence = item.get('confidence', 0.0)
                if s and p and o and confidence >= AlphaConfig.MIN_LLM_CONFIDENCE_TO_INGEST:
                    if verbose: print(f"  LLM proposed: ({s}, {p}, {o}) with confidence {confidence:.2f}")
                    extracted_triples.append((s, p, o))
            
            # Deduplicate to prevent LLM repetition loops
            extracted_triples = list(set(extracted_triples))
            
            if not extracted_triples:
                print("  [WARN] LLM returned no beliefs.")
        else:
            print(f"Symbolic match: {extracted_triples}")

        # 3. Validate extracted triples before revision
        validated_triples = []
        for s, p, o in extracted_triples:
            if self._is_valid_triple(s, p, o, verbose=verbose):
                validated_triples.append((str(s), str(p), str(o)))

        if not validated_triples and extracted_triples:
            print("  [WARN] All extracted triples were filtered out by validation.")

        # 4. Belief Revision (Update Graph)
        newly_created_ids = []
        new_belief_content = {}
        for subj, pred, obj in validated_triples:
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
        
        if prune:
            self.memory.prune_graph() # Enforce bounds
        
        # Resting Period: Check if we need to consolidate (Prune & Merge)
        self.session_new_beliefs_count += len(newly_created_ids)
        # Only consolidate if pruning is enabled (i.e., not in a batch operation)
        if prune and self.session_new_beliefs_count >= AlphaConfig.CONSOLIDATION_THRESHOLD:
            print(f"\n[System] Resting period: Consolidating after {self.session_new_beliefs_count} new beliefs...")
            self.consolidate()
            self.session_new_beliefs_count = 0 # Reset counter

        return extracted_triples

    def batch_learn(self, text: str, verbose: bool = False, focus_topic: str = None) -> List[Tuple]:
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
            if current_len + len(chunk) > 1000:
                if current_batch:
                    triples = self.learn(" ".join(current_batch), verbose, focus_topic=focus_topic, prune=False)
                    all_triples.extend(triples)
                current_batch = [chunk]
                current_len = len(chunk)
            else:
                current_batch.append(chunk)
                current_len += len(chunk)
        
        # Process remaining
        if current_batch:
            triples = self.learn(" ".join(current_batch), verbose, focus_topic=focus_topic, prune=False)
            all_triples.extend(triples)
            
        self.memory.prune_graph() # Prune once at the end of the batch

        # Check for consolidation at the end of the batch operation.
        if self.session_new_beliefs_count >= AlphaConfig.CONSOLIDATION_THRESHOLD:
            print(f"\n[System] Resting period: Consolidating after batch learning ({self.session_new_beliefs_count} new beliefs)...")
            self.consolidate()
            self.session_new_beliefs_count = 0 # Reset counter

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
        active_context_unfiltered = [b for b in raw_context if b['confidence'] >= AlphaConfig.MIN_CONFIDENCE_FOR_CONTEXT]
        
        # Limit the number of beliefs passed to the reasoner based on the mode.
        limit = AlphaConfig.TOP_K_DEEP_THINK if self.deep_think_mode else AlphaConfig.TOP_K_ACTIVATION
        active_context = active_context_unfiltered[:limit]
        print(f"Activated {len(active_context)} beliefs (total context: {len(raw_context)}).")
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

        # Optimization: Use lightweight synthesis if we have matches AND we are NOT in deep think mode.
        # If deep_think is True, we skip this to force the richer 'reason' path below.
        if matches and not self.deep_think_mode:
            # The previous template-based response was too simplistic for general queries.
            # We now always use the synthesizer in default mode to provide a richer,
            # more natural-sounding answer based on multiple relevant beliefs.
            print("Found relevant beliefs. Synthesizing a concise answer...")
            selected = matches # `matches` is already derived from the limited `active_context`
            response = self.llm.synthesize(user_input, selected, verbose=verbose)
            if response:
                return response


        # 6. LLM Reasoner (Fallback)
        if not response:
            # Optimization: If we know nothing relevant, don't hallucinate or waste tokens.
            if not active_context:
                return "I don't have enough information in my memory to answer that."

            print("Deterministic logic insufficient. Calling LLM Reasoner...")
            # Use a smarter model if deep_think is enabled
            reasoning_model = "llama-3.3-70b-versatile" if self.deep_think_mode else "llama-3.1-8b-instant"
            self.current_interaction_llm_calls += 1
            response = self.llm.reason(user_input, active_context, verbose=verbose, model=reasoning_model)

        return response

    def process_interaction(self, user_input: str, verbose: bool = False, deep_think: bool = None):
        """
        Conversational wrapper: Decides whether to learn, then responds.
        Can temporarily override deep_think_mode for a single interaction.
        """
        original_deep_think_mode = self.deep_think_mode
        if deep_think is not None:
            self.deep_think_mode = deep_think
            if verbose:
                print(f"[System] Deep Think mode temporarily set to {self.deep_think_mode}")

        try:
            self.current_interaction_llm_calls = 0 # Reset for this interaction

            # Command: Explicit Research
            # Allows user to say "Learn about football" or "Research quantum physics"
            research_commands = ("learn about ", "research ", "force research ", "resarch ", "wiki ", "search about ", "search ")
            if user_input.lower().startswith(research_commands):
                is_force = user_input.lower().startswith("force research ")
                is_wiki = user_input.lower().startswith("wiki ")
                # The regex needs to match the longest command first, so we put `search about` before `search`.
                topic = re.sub(r"^(force research|learn about|research|resarch|wiki|search about|search)\s+", "", user_input, flags=re.IGNORECASE).strip()
                source = "wiki" if is_wiki else "web"
                self.research_topic(topic, verbose, force=is_force, source=source)
                return f"I have finished researching '{topic}' via {source} and updated my memory."

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
        finally:
            # Restore original deep think mode
            self.deep_think_mode = original_deep_think_mode

    def consolidate(self):
        """
        Maintenance task: Looks for redundant beliefs and merges them.
        """
        print("\n--- Running Consolidation Tick ---")
        
        # 1. Check Memory Pressure
        all_beliefs = self.memory.get_all_beliefs() or []
        capacity_usage = len(all_beliefs) / AlphaConfig.MAX_BELIEFS
        
        # 2. Pruning (Cleanup & Decay) - Only under pressure
        # Strategy: Capacity-Based Decay. Only prune when under memory pressure.
        # This ensures we retain knowledge indefinitely unless we need space.
        if capacity_usage > AlphaConfig.MEMORY_PRESSURE_THRESHOLD:
            print(f"[System] Memory pressure high ({capacity_usage:.1%}). Cleaning up and applying decay.")
            self.memory.forget_weak_beliefs()
            if hasattr(self.memory, 'apply_decay'):
                self.memory.apply_decay()
            else:
                print("[System] Warning: 'apply_decay' not found in MemoryStore (skipping).")

        # 3. Recompute Structural Support
        self.memory.recompute_structural_support()
        
        # 4. Fetch a batch of beliefs for merge consideration.
        # Fetching all beliefs is not scalable. Instead, fetch a sample of
        # recently added or frequently accessed beliefs.
        # For this example, we'll fetch the 100 most recently updated beliefs.
        if hasattr(self.memory, 'get_beliefs'):
            candidate_beliefs = self.memory.get_beliefs(sort_by='last_accessed', limit=100)
        else:
            # Fallback: Sort in memory using all_beliefs from step 2
            candidate_beliefs = sorted(all_beliefs, key=lambda x: x.get('last_accessed', x.get('id', 0)), reverse=True)[:100]

        if len(candidate_beliefs) < 2: return
        
        # 4. Ask LLM for merges
        self.current_interaction_llm_calls += 1 # Tracking maintenance calls too
        merges = self.llm.suggest_merges(candidate_beliefs[:AlphaConfig.TOP_K_DEEP_THINK]) # Use a configurable limit
        
        # 5. Apply merges
        for m in merges:
            print(f"Merging {m['merge_id']} -> {m['keep_id']}")
            self.memory.merge_beliefs(m['keep_id'], m['merge_id'])
            
        # 6. Check Curiosity
        self.evaluate_curiosity_trigger()

    def _create_structured_goals(self, root_topic: str, sub_topics: list, verbose: bool, parent_goal_id: int = None):
        """
        Recursively creates a hierarchy of research goals from a plan.
        This assumes `llm.plan_research` returns a list of dicts,
        e.g., [{'topic': '...', 'sub_topics': [...]}, ...],
        and that `memory.add_goal` accepts a `parent_id`.
        """
        for sub_topic_item in sub_topics:
            # Handle both flat lists of strings and structured dictionaries for resilience
            if isinstance(sub_topic_item, str):
                topic_name = sub_topic_item
                nested_sub_topics = []
            elif isinstance(sub_topic_item, dict):
                topic_name = sub_topic_item.get('topic')
                nested_sub_topics = sub_topic_item.get('sub_topics', [])
            else:
                if verbose: print(f"[Research] Skipping malformed plan item: {sub_topic_item}")
                continue

            if not topic_name:
                continue

            # Ensure the planned sub-topic is relevant to the original query
            if not self.llm.check_relevance(root_topic, topic_name):
                if verbose:
                    print(f"[Research] Pruning irrelevant sub-topic from plan: '{topic_name}' (not relevant to '{root_topic}')")
                continue

            # Create the goal. We assume `add_goal` is modified to accept `parent_id`.
            goal_desc = f"Research {topic_name}"
            g_id = self.memory.add_goal(goal_desc, priority=5, parent_id=parent_goal_id)
            if verbose: print(f"[Goal] Created goal: {goal_desc}")

            if nested_sub_topics:
                self._create_structured_goals(
                    root_topic=root_topic,
                    sub_topics=nested_sub_topics,
                    verbose=verbose,
                    parent_goal_id=g_id
                )

    def research_topic(self, topic: str, depth: int = 0, verbose: bool = False, root_topic: str = None, is_autonomous: bool = False, start_time: float = None, force: bool = False, source: str = "web", plan_first: bool = True):
        """
        Actively searches Wikipedia for a topic and learns from the summary.
        Recursively explores interesting sub-topics found during research.
        Respects a time budget to prevent non-stop processing.
        
        Args:
            root_topic: The original query (e.g. "Spain") to keep research focused.
            is_autonomous: If True, allows drifting by creating new goals for unrelated sparks.
            start_time: Timestamp of when the top-level research began.
            plan_first: If True, creates a research plan and sub-goals instead of executing immediately.
        """
        # --- Structural Research (Planning First) ---
        # If this is a top-level call and planning is requested, we plan first.
        # This applies to both user commands and autonomous goals (if they are "Research" goals).
        if depth == 0 and plan_first:
            if not root_topic: root_topic = topic
            print(f"[Research] Planning a structured deep dive for '{topic}'...")
            
            sub_topics = self.llm.plan_research(topic)
            
            # Create a parent goal for the overarching topic (container)
            # We use a slightly different description to avoid re-triggering planning for the root.
            root_goal_id = self.memory.add_goal(f"Deep Dive {root_topic}", priority=7)
            
            # The first sub-goal is to execute the research on the main topic itself.
            # We use "Execute Research" to signal that this is the action phase, not planning.
            self.memory.add_goal(f"Execute Research {topic}", priority=6, parent_id=root_goal_id)
            
            if sub_topics:
                print(f"[Research] Plan created: {sub_topics}. Creating sub-goals...")
                self._create_structured_goals(root_topic, sub_topics, verbose, parent_goal_id=root_goal_id)
            
            print(f"[Research] Plan and goals for '{topic}' have been created.")
            
            if not is_autonomous:
                print(f"[Research] Interactive Mode: Executing main research goal immediately...")
                self.research_topic(topic, depth=0, verbose=verbose, root_topic=root_topic, is_autonomous=is_autonomous, start_time=start_time, force=force, source=source, plan_first=False)
            else:
                print(f"[Research] Autonomous discovery will proceed with these goals.")
            
            return # The planning is done. The autonomous loop will execute the created goals.

        if start_time is None:
            start_time = time.time()

        # Check Knowledge Base (Deduplication)
        if not force:
            # Deduplicate based on the intent (Planning vs Execution)
            check_desc = f"Research {topic}" if plan_first else f"Execute Research {topic}"
            
            if self.memory.is_goal_achieved(check_desc):
                if verbose: 
                    print(f"[Research] Skipping '{topic}': Goal '{check_desc}' already achieved.")
                return

        # Check Time Budget
        time_limit = AlphaConfig.RESEARCH_MAX_TIME_DEEPTHINK_SECONDS if self.deep_think_mode else AlphaConfig.RESEARCH_MAX_TIME_SECONDS
        elapsed = time.time() - start_time
        if not AlphaConfig.RESEARCH_IGNORE_TIME_LIMIT:
            if elapsed > time_limit:
                if verbose: print(f"[Research] ⏳ Time budget exceeded ({elapsed:.1f}s / {time_limit}s). Deferring '{topic}' to future goals.")
                # Save the current topic as a goal to resume later
                self.memory.add_goal(f"Research {topic}", priority=5)
                return

        if depth > 2: # Limit recursion to prevent infinite rabbit holes
            return
            
        import tools
        
        learned_triples = []
        learned_something = False

        # 1. Web Search (Default)
        if source == "web":
            print(f"\n[Research] Searching Web for: '{topic}'...")
            max_res = 15 if self.deep_think_mode else 5
            results = tools.search_web(topic, max_results=max_res)
            
            if results:
                print(f"[Research] Found {len(results)} web results.")

                source_limit = AlphaConfig.RESEARCH_MAX_SOURCES_DEEPTHINK if self.deep_think_mode else AlphaConfig.RESEARCH_MAX_SOURCES
                processed_sources = 0

                for res in results:
                    # Re-check time budget inside the loop to stop processing mid-batch
                    elapsed = time.time() - start_time
                    if not AlphaConfig.RESEARCH_IGNORE_TIME_LIMIT and elapsed > time_limit:
                        if verbose: print(f"[Research] ⏳ Time budget exceeded during source processing ({elapsed:.1f}s / {time_limit}s). Stopping.")
                        break

                    if processed_sources >= source_limit:
                        if verbose: print(f"[Research] Source limit ({source_limit}) reached. Stopping.")
                        break

                    if not self.llm.check_search_result_relevance(topic, res['title'], res.get('summary') or ""):
                        if verbose: print(f"  [Research] Skipping irrelevant: {res['title']}")
                        continue

                    processed_sources += 1
                    print(f"  - Reading ({processed_sources}/{source_limit}): {res['title']}")
                    
                    # Fetch full page content for better learning
                    page_text = tools.fetch_webpage_text(res['link'])
                    if len(page_text) > 500:
                        content = f"Source: {res['title']}\nContent: {page_text}"
                    else:
                        content = f"{res['title']}: {res['summary']}"
                        
                    focus = root_topic if root_topic else topic
                    triples = self.batch_learn(content, verbose=verbose, focus_topic=focus)
                    learned_triples.extend(triples)
                learned_something = True
            else:
                print("[Research] No web results found. Falling back to Wikipedia.")
                source = "wiki" # Fallback

        # 2. Wikipedia Search (Fallback or Explicit)
        if source == "wiki":
            print(f"\n[Research] Searching Wikipedia for: '{topic}'...")
            url, summary = tools.search_wikipedia(topic)
            
            if url and summary:
                print(f"[Research] Found: {url}")
                print(f"[Research] Reading summary ({len(summary)} chars)...")
                snippet = summary[:AlphaConfig.MAX_CHUNK_SIZE] 
                focus = root_topic if root_topic else topic
                triples = self.batch_learn(snippet, verbose=verbose, focus_topic=focus)
                learned_triples.extend(triples)
                learned_something = True
            else:
                print("[Research] No Wikipedia results found.")

        if not learned_something:
            return
        
        # --- Serendipitous Curiosity ---
        # If we are in focused mode (not autonomous) and we are deep in recursion (depth > 0),
        # we assume the plan covers what is needed. We stop here to prevent drift.
        if not is_autonomous and depth > 0:
            return

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
                    # Do not pass start_time, so each sub-topic gets its own fresh time budget.
                    # The depth limit prevents infinite recursion.
                    self.research_topic(spark, depth=depth + 1, verbose=verbose, root_topic=root_topic, is_autonomous=False, source=source)
                else:
                    if verbose: print(f"[Research] Focused: Ignoring '{spark}' (tangential).")

    def read_news(self, verbose: bool = False):
        """
        Fetches latest news, learns from it, and performs deep research on interesting/unknown topics.
        """
        try:
            import tools
        except ImportError:
            print("[System] Tools module not found.")
            return

        print("\n--- 📰 Scanning Global News Feeds ---")
        news_items = tools.fetch_rss_news()
        
        if not news_items:
            print("[News] No items found or connection failed.")
            return

        # Limit processing to avoid overload
        process_count = min(len(news_items), AlphaConfig.MAX_NEWS_HEADLINES)
        print(f"[News] Retrieved {len(news_items)} headlines. Processing top {process_count}...")
        
        research_limit = 2
        research_count = 0
        
        for i, item in enumerate(news_items[:process_count]):
            headline = item['title']
            summary = item.get('summary', '')
            content = f"{headline}. {summary}"
            
            print(f"\n[{i+1}/{len(news_items)}] Learning: {headline}")
            
            # 1. Fast Learn (Surface)
            triples = self.learn(content, verbose=verbose, prune=False)
            
            # 2. Check Curiosity
            sparks = self._identify_curiosity_sparks(triples)
            
            if sparks and research_count < research_limit:
                research_count += 1
                # Prioritize the spark that looks most like a specific entity (Capitalized, longer)
                target = max(sparks, key=len)
                
                print(f"[News] 💡 Curiosity triggered: I don't know much about '{target}'.")
                print(f"[News] Launching investigation into '{target}'...")
                
                # Trigger research
                self.research_topic(target, depth=0, verbose=verbose, source="web")
            elif sparks:
                target = max(sparks, key=len)
                print(f"[News] 💡 Curiosity triggered for '{target}', but research limit reached. Adding as goal.")
                self.memory.add_goal(f"Research {target}", priority=3)
            else:
                if verbose: print("[News] Concepts appear familiar. Moving on.")
        
        # Cleanup after batch
        self.memory.prune_graph()
        # Check for consolidation at the end of the news session.
        if self.session_new_beliefs_count >= AlphaConfig.CONSOLIDATION_THRESHOLD:
            print(f"\n[System] Resting period: Consolidating after news session ({self.session_new_beliefs_count} new beliefs)...")
            self.consolidate()
            self.session_new_beliefs_count = 0 # Reset counter

        print("\n[News] Session complete.")

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
                        # "Research" implies we should plan it out first.
                        self.research_topic(topic, verbose=verbose, is_autonomous=True, plan_first=True)
                        self.memory.complete_goal(next_goal['id'])
                        print(f"[Goal] Marked '{next_goal['description']}' as achieved.")
                    
                    elif next_goal['description'].startswith("Execute Research "):
                        topic = next_goal['description'].replace("Execute Research ", "")
                        # "Execute Research" implies the planning is done, just do the work.
                        self.research_topic(topic, verbose=verbose, is_autonomous=True, plan_first=False)
                        self.memory.complete_goal(next_goal['id'])
                        print(f"[Goal] Marked '{next_goal['description']}' as achieved.")

                    elif next_goal['description'].startswith("Deep Dive "):
                        # Container goal for a research plan. Mark as achieved to unblock sub-goals.
                        self.memory.complete_goal(next_goal['id'])
                        print(f"[Goal] Processed container: '{next_goal['description']}'.")
                    
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

    def _is_valid_triple(self, s: str, p: str, o: str, verbose: bool = False) -> bool:
        """Performs basic sanity checks on an extracted triple."""
        # Check for empty parts
        if not all((s, p, o)):
            if verbose: print(f"  [INVALID] Triple part is empty: ({s}, {p}, {o})")
            return False

        # Convert to string now that we know they are not None/empty/0, to prevent TypeErrors.
        s, p, o = str(s), str(p), str(o)

        # Check for excessive length (could be a sign of poor extraction)
        # Assumes AlphaConfig.MAX_TRIPLE_PART_LENGTH exists
        max_len = getattr(AlphaConfig, 'MAX_TRIPLE_PART_LENGTH', 100)
        if any(len(part) > max_len for part in [s, p, o]):
            if verbose: print(f"  [INVALID] Triple part too long: ({s}, {p}, {o})")
            return False

        # Check if subject or object are just stop words (after lowercasing)
        # Assumes AlphaConfig.STOP_WORDS exists
        stop_words = getattr(AlphaConfig, 'STOP_WORDS', set())
        if s.lower() in stop_words or o.lower() in stop_words:
            if verbose: print(f"  [INVALID] Subject or object is a stopword: ({s}, {p}, {o})")
            return False

        return True

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
        # The previous logic incorrectly discarded non-capitalized keywords if any capitalized words
        # were present (e.g., in "caste system in India", "caste system" would be ignored).
        # The new logic correctly tokenizes the entire input string to gather all potential keywords,
        # ensuring a more comprehensive context is retrieved.
        all_words = re.findall(r'\w+', text.lower())
        tokens = [t for t in all_words if t not in AlphaConfig.STOP_WORDS and len(t) > 3]
        
        # Filter out generic entity types (e.g. "system", "group") to prevent context pollution
        # unless they are the only tokens available.
        refined_tokens = [t for t in tokens if t not in AlphaConfig.COMMON_ENTITIES]
        if refined_tokens:
            tokens = refined_tokens
        
        if not tokens: return []
        
        # Use the Activation Engine (PageRank / Spreading Activation)
        try:
            # The ActivationEngine provides superior context through graph-based activation spreading.
            # We use it for both modes, simply adjusting the number of results returned.
            if self.deep_think_mode:
                # For deep think, use the activation engine but request a larger, richer context.
                limit = AlphaConfig.TOP_K_DEEP_THINK
                return self.activation.get_activated_beliefs(tokens, limit=limit)
            else:
                # For normal mode, the sophisticated (but smaller) context from the
                # ActivationEngine is used with its default limit.
                return self.activation.get_activated_beliefs(tokens)
        except Exception as e:
            print(f"[System] Warning: Context retrieval failed (skipping): {e}")
            return []

    def close(self):
        self.memory.close()