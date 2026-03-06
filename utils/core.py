import re
import time
import random
from collections import deque
from typing import List, Tuple, Dict, Any
from utils.storage import MemoryStore
from utils.extractors import SymbolicExtractor, LLMExtractor
from utils.config import AlphaConfig
from utils.confidence import compute_confidence
from utils.activation import ActivationEngine
from rapidfuzz import process, fuzz
import utils.tools as tools
 
class PECSCore:
    def __init__(self, groq_api_key: str, deep_think_mode: bool = False, event_handler=None):
        self.memory = MemoryStore()
        self.event_handler = event_handler

        # --- Data Integrity Check ---
        if hasattr(self.memory, 'clean_orphaned_edges'):
            self.memory.clean_orphaned_edges()

        # OPTIMIZATION: Initialize Spacy Extractor
        from utils.extractors import SpacyExtractor, SymbolicExtractor
        self.spacy_extractor = SpacyExtractor()
        
        self.symbolic = SymbolicExtractor()
        self.llm = LLMExtractor(groq_api_key, token_callback=self._handle_token_update)
        
        self.deep_think_mode = deep_think_mode
        self.activation = ActivationEngine(self.memory)
        
        # Resource Awareness & Curiosity
        self.llm_call_history = deque(maxlen=AlphaConfig.LLM_BUDGET_WINDOW)
        self.curiosity_goals_history = deque(maxlen=AlphaConfig.MAX_CURIOSITY_GOALS_WINDOW)
        self.current_interaction_llm_calls = 0
        self.session_new_beliefs_count = 0
        self.interaction_logs = [] # Capture logs for the frontend


    def _correct_tokens(self, tokens):
        """
        Attempts fuzzy correction of tokens against known belief subjects.
        """
        beliefs = self.memory.get_all_beliefs()
        subjects = list({b['subject'] for b in beliefs if b.get('subject')})

        corrected = []

        for token in tokens:
            match = process.extractOne(token, subjects, scorer=fuzz.ratio)
            if match and match[1] > 85:
                corrected.append(match[0])
            else:
                corrected.append(token)

        return corrected

    def log(self, message: str):
        """Logs a message to both console and the interaction log list."""
        print(message)
        self.interaction_logs.append(str(message))
        
    # --- NEW METHOD ---
    def _handle_token_update(self, total_tokens: int):
        """Passes token usage from LLMExtractor to the server via event handler."""
        if self.event_handler:
            self.event_handler('token_update', {'total': total_tokens})
        
        
    def learn(self, user_input: str, verbose: bool = False, focus_topic: str = None, prune: bool = True) -> List[Tuple]:
        """
        Ingests knowledge from input using optimized Spacy > Symbolic > LLM pipeline.
        """
        # Clean and truncate input for logging
        clean_input = ' '.join(user_input.split())
        truncate_at = 70
        truncated_input = (clean_input[:truncate_at] + '...') if len(clean_input) > truncate_at else clean_input
        self.log(f"\n--- Learning from: '{truncated_input}' ---")
        
        # Always get context first to establish relationships for new beliefs
        raw_context = self._get_relevant_context(user_input)
        context = raw_context

        # OPTIMIZATION: 1. Spacy Extraction (Primary Local Method)
        if verbose: self.log("Attempting Spacy extraction...")
        extracted_triples = self.spacy_extractor.process(user_input)
        
        # 2. Symbolic Regex (Secondary Local Method)
        if not extracted_triples:
            if verbose: self.log("Spacy failed. Attempting Symbolic Regex...")
            extracted_triples = self.symbolic.process(user_input)
        
        # 3. Fallback to LLM Extraction (if local methods failed)
        if not extracted_triples:
            self.log("Local extraction ambiguous. Calling LLM...")
            self.current_interaction_llm_calls += 1
            llm_result = self.llm.extract(user_input, context, verbose=verbose, focus_topic=focus_topic)
            
            extracted_triples = []
            for item in llm_result.get("proposed_beliefs", []):
                s = item.get('subject')
                p = item.get('predicate')
                o = item.get('object')
                confidence = item.get('confidence', 0.0)
                if s and p and o and confidence >= AlphaConfig.MIN_LLM_CONFIDENCE_TO_INGEST:
                    if verbose: self.log(f"  LLM proposed: ({s}, {p}, {o}) with confidence {confidence:.2f}")
                    extracted_triples.append((str(s), str(p), str(o)))
            
            extracted_triples = list(set(extracted_triples))
            
            if not extracted_triples:
                self.log("  [WARN] LLM returned no beliefs.")
        else:
            self.log(f"Local Match: {extracted_triples}")

        # 4. Validate extracted triples before revision
        validated_triples = []
        for s, p, o in extracted_triples:
            if self._is_valid_triple(s, p, o, verbose=verbose):
                validated_triples.append((str(s), str(p), str(o)))

        if not validated_triples and extracted_triples:
            self.log("  [WARN] All extracted triples were filtered out by validation.")

        # 5. Belief Revision (Update Graph)
        newly_created_ids = []
        new_belief_content = {}
        contradiction_candidates = [] 
        
        for subj, pred, obj in validated_triples:
            # Normalize
            subj, pred, obj = subj.lower(), pred.lower(), obj.lower()
            
            # Add/Update Belief
            b_id, is_new = self.memory.add_belief(subj, pred, obj)
            action = "Created" if is_new else "Strengthened"
            self.log(f"{action} belief: ({subj}, {pred}, {obj})")
            
            if is_new:
                # GATHER potential conflicts
                potential_conflicts = self.memory.get_potential_conflicts(subj, pred)
                for pc in potential_conflicts:
                    if pc['id'] != b_id and pc['object'] != obj:
                        contradiction_candidates.append({
                            'new_id': b_id,
                            'existing_id': pc['id'],
                            'fact1': (subj, pred, obj),
                            'fact2': (pc['subject'], pc['predicate'], pc['object'])
                        })

                if self.event_handler:
                    self.event_handler('belief_created', {
                        'id': b_id, 'subject': subj, 'predicate': pred, 'object': obj,
                        'evidence_score': 1.0, 'usage_count': 0
                    })
                newly_created_ids.append(b_id)
                new_belief_content[b_id] = (subj, obj)
            else:
                if self.event_handler:
                    self.event_handler('belief_strengthened', {'id': b_id})

        # --- EXECUTE BATCHED CONTRADICTION CHECK ---
        if contradiction_candidates:
            self.current_interaction_llm_calls += 1
            actual_contradictions = self.llm.check_contradictions_batch(contradiction_candidates, verbose=verbose)
            
            for pair in actual_contradictions:
                edge_id = self.memory.mark_contradiction(pair['new_id'], pair['existing_id'])
                if self.event_handler and edge_id:
                    self.event_handler('edge_created', {
                        'id': edge_id, 'source_id': pair['new_id'], 
                        'target_id': pair['existing_id'], 'type': 'contradicts'
                    })

        # 6. Semantic Linking (Context & Rooting)
        new_edges_events = []
        
        for new_id in newly_created_ids:
            n_subj, n_obj = new_belief_content[new_id]
            n_tokens = set(re.findall(r'\w+', f"{n_subj} {n_obj}".lower())) - AlphaConfig.STOP_WORDS
            
            for context_belief in context:
                if new_id != context_belief['id']:
                    c_tokens = set(re.findall(r'\w+', f"{context_belief['subject']} {context_belief['object']}".lower())) - AlphaConfig.STOP_WORDS
                    
                    if not n_tokens.isdisjoint(c_tokens):
                        edge_id = self.memory.add_edge(source_id=context_belief['id'], target_id=new_id, edge_type='related_to', weight=0.5)
                        new_edges_events.append({
                            'id': edge_id, 'source_id': context_belief['id'], 
                            'target_id': new_id, 'type': 'related_to'
                        })

        if focus_topic and newly_created_ids:
            focus_tokens = [t for t in re.findall(r'\w+', focus_topic.lower()) if t not in AlphaConfig.STOP_WORDS]
            if focus_tokens:
                root_candidates = self.memory.search_beliefs(focus_tokens, limit=1)
                if root_candidates:
                    root_id = root_candidates[0]['id']
                    for new_id in newly_created_ids:
                        if new_id != root_id:
                            edge_id = self.memory.add_edge(source_id=root_id, target_id=new_id, edge_type='has_context', weight=0.8)
                            new_edges_events.append({
                                'id': edge_id, 'source_id': root_id, 
                                'target_id': new_id, 'type': 'has_context'
                            })
        
        self.memory.conn.commit()
        
        if self.event_handler:
            for event in new_edges_events:
                self.event_handler('edge_created', event)

        if prune:
            self.memory.prune_graph() 
        
        self.session_new_beliefs_count += len(newly_created_ids)
        if prune and self.session_new_beliefs_count >= AlphaConfig.CONSOLIDATION_THRESHOLD:
            self.log(f"\n[System] Resting period: Consolidating after {self.session_new_beliefs_count} new beliefs...")
            self.consolidate(verbose=verbose)
            self.session_new_beliefs_count = 0 

        return extracted_triples


    def batch_learn(self, text: str, verbose: bool = False, focus_topic: str = None) -> List[Tuple]:
        """
        Splits long text into chunks and learns from them sequentially.
        Efficient for ingesting news articles or documents.
        """
        self.log(f"\n--- Batch Learning ({len(text)} chars) ---")
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
            self.log(f"\n[System] Resting period: Consolidating after batch learning ({self.session_new_beliefs_count} new beliefs)...")
            self.consolidate(verbose=verbose)
            self.session_new_beliefs_count = 0 # Reset counter

        return all_triples

    def query(self, user_input: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Retrieves context and generates a response. 
        OPTIMIZATION: Uses deterministic pathfinding for relational questions.
        """
        self.log(f"\n--- Querying: '{user_input}' ---")
        
        reasoning_trace = None
        
        # --- OPTIMIZATION: Deterministic Pathfinding ---
        # Checks if user is asking about connection/relationship between two things
        # This saves LLM tokens and provides exact graph-based reasoning
        lower_input = user_input.lower()
        if "relate" in lower_input or "connect" in lower_input or "link" in lower_input or "between" in lower_input:
            import networkx as nx
            
            # Use Spacy from our extractor to find noun chunks
            doc = self.spacy_extractor.nlp(user_input)
            # Filter stopwords
            entities = [chunk.text for chunk in doc.noun_chunks if chunk.text.lower() not in AlphaConfig.STOP_WORDS]
            
            if len(entities) >= 2:
                # Take the two longest entities found (most specific)
                entities.sort(key=len, reverse=True)
                target_a = entities[0]
                target_b = entities[1]
                
                self.log(f"[Pathfinding] Checking connection between: '{target_a}' and '{target_b}'")
                
                start_nodes = self.memory.search_beliefs([target_a], limit=1)
                end_nodes = self.memory.search_beliefs([target_b], limit=1)
                
                if start_nodes and end_nodes:
                    start_id = start_nodes[0]['id']
                    end_id = end_nodes[0]['id']
                    
                    # Build a temporary graph of the neighborhoods of both nodes
                    # This allows finding a path even if they aren't directly connected
                    nodes_a, edges_a = self.memory.get_subgraph([start_id])
                    nodes_b, edges_b = self.memory.get_subgraph([end_id])
                    
                    # Merge into a transient NetworkX graph
                    G = nx.DiGraph()
                    all_nodes = {n['id']: n for n in nodes_a + nodes_b}
                    all_edges = edges_a + edges_b
                    
                    for nid, n in all_nodes.items():
                        label = n.get('object') or n.get('subject')
                        G.add_node(nid, label=label)
                        
                    for e in all_edges:
                        G.add_edge(e['source_id'], e['target_id'], type=e['type'])
                    
                    try:
                        path = nx.shortest_path(G, source=start_id, target=end_id)
                        
                        # Trace found! Construct explanation
                        steps = []
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i+1]
                            u_lbl = G.nodes[u]['label']
                            v_lbl = G.nodes[v]['label']
                            edge_data = G.get_edge_data(u, v)
                            pred = edge_data['type']
                            steps.append(f"{u_lbl} --[{pred}]--> {v_lbl}")
                            
                        path_str = " -> ".join(steps)
                        
                        # Visually highlight the path
                        if self.event_handler:
                            path_data = [{'id': pid, 'score': 1.0} for pid in path]
                            self.event_handler('beliefs_activated', path_data)

                        return {
                            "answer": f"I found a direct connection in my memory:\n{path_str}",
                            "reasoning": "Graph pathfinding algorithm found a direct path.",
                            "logs": self.interaction_logs
                        }
                    
                    except nx.NetworkXNoPath:
                        self.log("[Pathfinding] No path found in immediate neighborhood.")
                    except Exception as e:
                        self.log(f"[Pathfinding] Error: {e}")

        # 4. Standard Activation (Original Logic)
        raw_context = self._get_relevant_context(user_input)
        
        # --- Autonomous Self-Correction ---
        if not raw_context:
            proper_nouns = re.findall(r'\b[A-Z][a-zA-Z0-9]+\b', user_input)
            keywords = [t for t in proper_nouns if t.lower() not in AlphaConfig.STOP_WORDS and len(t) > 3]
            
            if keywords:
                keywords.sort(key=len, reverse=True)
                topic_to_research = keywords[0]
                self.log(f"[Self-Correction] Unknown topic '{topic_to_research}'. Researching...")
                self.research_topic(topic_to_research, verbose=verbose)
                raw_context = self._get_relevant_context(user_input)
        
        active_context_unfiltered = [b for b in raw_context if b['confidence'] >= AlphaConfig.MIN_CONFIDENCE_FOR_CONTEXT]
        limit = AlphaConfig.TOP_K_DEEP_THINK if self.deep_think_mode else AlphaConfig.TOP_K_ACTIVATION
        active_context = active_context_unfiltered[:limit]
        
        if self.event_handler:
            activated_data = [{'id': b['id'], 'score': b.get('relevance_score', 0)} for b in active_context]
            self.event_handler('beliefs_activated', activated_data)
            
        self.log(f"Activated {len(active_context)} beliefs.")

        # 5. Planning / Response
        response = None
        matches = []
        causal_terms = ["due to", "because", "caused", "reason", "fueling", "shaping", "linked", "tied", "driven", "result", "leads to"]

        stop_words = AlphaConfig.STOP_WORDS
        input_words = re.findall(r'\w+', user_input.lower())
        clean_input_tokens = {t for t in input_words if t not in stop_words}

        for b in active_context:
            is_causal = any(term in b['predicate'] for term in causal_terms)
            b_content = set(re.findall(r'\w+', (b['subject'] or "").lower())) | set(re.findall(r'\w+', (b['object'] or "").lower()))
            is_relevant = not clean_input_tokens.isdisjoint(b_content)
            
            if "why" in user_input.lower():
                if is_causal and is_relevant:
                    matches.append(b)
            elif b['confidence'] > 0.75 and is_relevant:
                matches.append(b)

        if matches and not self.deep_think_mode:
            self.log("Found relevant beliefs. Synthesizing a concise answer...")
            selected = matches 
            response = self.llm.synthesize(user_input, selected, verbose=verbose)
            if response:
                return {
                    "answer": response,
                    "reasoning": "Synthesized from high-confidence beliefs.",
                    "logs": self.interaction_logs
                }

        # 6. LLM Reasoner (Fallback)
        if not response:
            if not active_context:
                return {
                    "answer": "I don't have enough information in my memory to answer that.",
                    "reasoning": None,
                    "logs": self.interaction_logs
                }

            self.log("Deterministic logic insufficient. Calling LLM Reasoner...")
            reasoning_model = AlphaConfig.REASONING_MODEL_DEEP if self.deep_think_mode else AlphaConfig.REASONING_MODEL_FAST
            self.current_interaction_llm_calls += 1
            reason_result = self.llm.reason(user_input, active_context, verbose=verbose, model=reasoning_model)
            response = reason_result['answer']
            reasoning_trace = reason_result.get('reasoning')
            
            if self.event_handler and reasoning_trace:
                self.event_handler('reasoning', reasoning_trace)

        return {
            "answer": response,
            "reasoning": reasoning_trace,
            "logs": self.interaction_logs
        }
    

    def process_interaction(self, user_input: str, verbose: bool = False, deep_think: bool = None) -> Dict[str, Any]:
        """
        Conversational wrapper: Decides whether to learn, then responds.
        Can temporarily override deep_think_mode for a single interaction.
        """
        original_deep_think_mode = self.deep_think_mode
        self.interaction_logs = [] # Reset logs for new interaction
        
        if deep_think is not None:
            self.deep_think_mode = deep_think
            if verbose:
                self.log(f"[System] Deep Think mode temporarily set to {self.deep_think_mode}")

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

                # Emit a status message to the UI before starting the long-running task
                if self.event_handler:
                    self.event_handler('system_message', {'message': f"Beginning research on '{topic}' via {source}..."})

                self.research_topic(topic, verbose, force=is_force, source=source)
                return {
                    "answer": f"I have finished researching '{topic}' via {source} and updated my memory.",
                    "reasoning": None,
                    "logs": self.interaction_logs
                }

            # Optimization: If input is a question, skip extraction to save tokens
            is_question = "?" in user_input or user_input.lower().strip().startswith(
                ("what", "who", "where", "when", "why", "how", "tell", "describe", "explain", "show")
            )
            
            if not is_question:
                self.learn(user_input, verbose)
            else:
                self.log(f"\n--- Processing: '{user_input}' (Read-Only) ---")
                
            result = self.query(user_input, verbose)

            # Record usage and check curiosity
            self.llm_call_history.append(self.current_interaction_llm_calls)
            self.evaluate_curiosity_trigger(verbose)

            # Filter output based on modes
            answer = result.get("answer")
            reasoning = result.get("reasoning")
            logs = result.get("logs")

            if not self.deep_think_mode:
                reasoning = None

            if not verbose:
                logs = None

            return {
                "answer": answer,
                "reasoning": reasoning,
                "logs": logs
            }
        finally:
            # Restore original deep think mode
            self.deep_think_mode = original_deep_think_mode

    def consolidate(self, verbose: bool = False):
        """
        Maintenance task: Looks for redundant beliefs and merges them.
        """
        self.log("\n--- Running Consolidation Tick ---")
        
        # 1. Check Memory Pressure
        all_beliefs = self.memory.get_all_beliefs() or []
        capacity_usage = len(all_beliefs) / AlphaConfig.MAX_BELIEFS
        
        # 2. Pruning (Cleanup & Decay) - Only under pressure
        # Strategy: Capacity-Based Decay. Only prune when under memory pressure.
        # This ensures we retain knowledge indefinitely unless we need space.
        if capacity_usage > AlphaConfig.MEMORY_PRESSURE_THRESHOLD:
            self.log(f"[System] Memory pressure high ({capacity_usage:.1%}). Cleaning up and applying decay.")
            self.memory.forget_weak_beliefs()
            if hasattr(self.memory, 'apply_decay'):
                self.memory.apply_decay()
            else:
                self.log("[System] Warning: 'apply_decay' not found in MemoryStore (skipping).")

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
        merges = self.llm.suggest_merges(candidate_beliefs[:AlphaConfig.TOP_K_DEEP_THINK], verbose=verbose) # Use a configurable limit
        
        # 5. Apply merges
        for m in merges:
            self.log(f"Merging {m['merge_id']} -> {m['keep_id']}")
            self.memory.merge_beliefs(m['keep_id'], m['merge_id'])
            if self.event_handler:
                self.event_handler('belief_merged', m)
            
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
                if verbose: self.log(f"[Research] Skipping malformed plan item: {sub_topic_item}")
                continue

            if not topic_name:
                continue

            # Ensure the planned sub-topic is relevant to the original query
            if not self.llm.check_relevance(root_topic, topic_name, verbose=verbose):
                if verbose:
                    self.log(f"[Research] Pruning irrelevant sub-topic from plan: '{topic_name}' (not relevant to '{root_topic}')")
                continue

            # Create the goal. We assume `add_goal` is modified to accept `parent_id`.
            goal_desc = f"Research {topic_name}"
            g_id = self.memory.add_goal(goal_desc, priority=5, parent_id=parent_goal_id)
            if verbose: self.log(f"[Goal] Created goal: {goal_desc}")

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
            self.log(f"[Research] Planning a structured deep dive for '{topic}'...")
            
            sub_topics = self.llm.plan_research(topic, verbose=verbose)
            
            # Create a parent goal for the overarching topic (container)
            # We use a slightly different description to avoid re-triggering planning for the root.
            root_goal_id = self.memory.add_goal(f"Deep Dive {root_topic}", priority=7)
            
            # The first sub-goal is to execute the research on the main topic itself.
            # We use "Execute Research" to signal that this is the action phase, not planning.
            self.memory.add_goal(f"Execute Research {topic}", priority=6, parent_id=root_goal_id)
            
            if sub_topics:
                self.log(f"[Research] Plan created: {sub_topics}. Creating sub-goals...")
                self._create_structured_goals(root_topic, sub_topics, verbose, parent_goal_id=root_goal_id)
            
            self.log(f"[Research] Plan and goals for '{topic}' have been created.")
            
            if not is_autonomous:
                self.log(f"[Research] Interactive Mode: Executing main research goal immediately...")
                self.research_topic(topic, depth=0, verbose=verbose, root_topic=root_topic, is_autonomous=is_autonomous, start_time=start_time, force=force, source=source, plan_first=False)
            else:
                self.log(f"[Research] Autonomous discovery will proceed with these goals.")
            
            return # The planning is done. The autonomous loop will execute the created goals.

        if start_time is None:
            start_time = time.time()

        # Check Knowledge Base (Deduplication)
        if not force:
            # Deduplicate based on the intent (Planning vs Execution)
            check_desc = f"Research {topic}" if plan_first else f"Execute Research {topic}"
            
            if self.memory.is_goal_achieved(check_desc):
                if verbose: 
                    self.log(f"[Research] Skipping '{topic}': Goal '{check_desc}' already achieved.")
                return

        # Check Time Budget
        time_limit = AlphaConfig.RESEARCH_MAX_TIME_DEEPTHINK_SECONDS if self.deep_think_mode else AlphaConfig.RESEARCH_MAX_TIME_SECONDS
        elapsed = time.time() - start_time
        if not AlphaConfig.RESEARCH_IGNORE_TIME_LIMIT:
            if elapsed > time_limit:
                if verbose: self.log(f"[Research] ⏳ Time budget exceeded ({elapsed:.1f}s / {time_limit}s). Deferring '{topic}' to future goals.")
                # Save the current topic as a goal to resume later
                self.memory.add_goal(f"Research {topic}", priority=5)
                return

        if depth > 2: # Limit recursion to prevent infinite rabbit holes
            return
            
        import utils.tools as tools
        
        learned_triples = []
        learned_something = False

        # 1. Web Search (Default)
        if source == "web":
            self.log(f"\n[Research] Searching Web for: '{topic}'...")
            max_res = 15 if self.deep_think_mode else 5
            results = tools.search_web(topic, max_results=max_res)
            
            if results:
                self.log(f"[Research] Found {len(results)} web results.")

                source_limit = AlphaConfig.RESEARCH_MAX_SOURCES_DEEPTHINK if self.deep_think_mode else AlphaConfig.RESEARCH_MAX_SOURCES
                processed_sources = 0

                for res in results:
                    # Re-check time budget inside the loop to stop processing mid-batch
                    elapsed = time.time() - start_time
                    if not AlphaConfig.RESEARCH_IGNORE_TIME_LIMIT and elapsed > time_limit:
                        if verbose: self.log(f"[Research] ⏳ Time budget exceeded during source processing ({elapsed:.1f}s / {time_limit}s). Stopping.")
                        break

                    if processed_sources >= source_limit:
                        if verbose: self.log(f"[Research] Source limit ({source_limit}) reached. Stopping.")
                        break

                    if not self.llm.check_search_result_relevance(topic, res['title'], res.get('summary') or "", verbose=verbose):
                        if verbose: self.log(f"  [Research] Skipping irrelevant: {res['title']}")
                        continue

                    processed_sources += 1
                    self.log(f"  - Reading ({processed_sources}/{source_limit}): {res['title']}")
                    
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
                self.log("[Research] No web results found. Falling back to Wikipedia.")
                source = "wiki" # Fallback

        # 2. Wikipedia Search (Fallback or Explicit)
        if source == "wiki":
            self.log(f"\n[Research] Searching Wikipedia for: '{topic}'...")
            url, summary = tools.search_wikipedia(topic)
            
            if url and summary:
                self.log(f"[Research] Found: {url}")
                self.log(f"[Research] Reading summary ({len(summary)} chars)...")
                snippet = summary[:AlphaConfig.MAX_CHUNK_SIZE] 
                focus = root_topic if root_topic else topic
                triples = self.batch_learn(snippet, verbose=verbose, focus_topic=focus)
                learned_triples.extend(triples)
                learned_something = True
            else:
                self.log("[Research] No Wikipedia results found.")

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
                self.log(f"[Research] Autonomous: Found interesting spark '{spark}'. Adding to future goals.")
                self.memory.add_goal(f"Research {spark}", priority=5)
            else:
                # User Mode: Stick to the original query.
                # Only dive if strictly relevant to the root topic.
                if root_topic and self.llm.check_relevance(root_topic, spark, verbose=verbose):
                    self.log(f"[Research] Focused: '{spark}' is relevant to '{root_topic}'. Diving deeper...")
                    # Do not pass start_time, so each sub-topic gets its own fresh time budget.
                    # The depth limit prevents infinite recursion.
                    self.research_topic(spark, depth=depth + 1, verbose=verbose, root_topic=root_topic, is_autonomous=False, source=source)
                else:
                    if verbose: self.log(f"[Research] Focused: Ignoring '{spark}' (tangential).")

    def read_news(self, verbose: bool = False):
        """
        Fetches latest news, learns from it, and performs deep research on interesting/unknown topics.
        """
        try:
            import utils.tools as tools
        except ImportError:
            self.log("[System] Tools module not found.")
            return

        self.log("\n--- 📰 Scanning Global News Feeds ---")
        news_items = tools.fetch_rss_news()
        
        if not news_items:
            self.log("[News] No items found or connection failed.")
            return

        # Limit processing to avoid overload
        process_count = min(len(news_items), AlphaConfig.MAX_NEWS_HEADLINES)
        self.log(f"[News] Retrieved {len(news_items)} headlines. Processing top {process_count}...")
        
        research_limit = 2
        research_count = 0
        
        for i, item in enumerate(news_items[:process_count]):
            headline = item['title']
            summary = item.get('summary', '')
            content = f"{headline}. {summary}"
            
            self.log(f"\n[{i+1}/{len(news_items)}] Learning: {headline}")
            
            # 1. Fast Learn (Surface)
            triples = self.learn(content, verbose=verbose, prune=False)
            
            # 2. Check Curiosity
            sparks = self._identify_curiosity_sparks(triples)
            
            if sparks and research_count < research_limit:
                research_count += 1
                # Prioritize the spark that looks most like a specific entity (Capitalized, longer)
                target = max(sparks, key=len)
                
                self.log(f"[News] 💡 Curiosity triggered: I don't know much about '{target}'.")
                self.log(f"[News] Launching investigation into '{target}'...")
                
                # Trigger research
                self.research_topic(target, depth=0, verbose=verbose, source="web")
            elif sparks:
                target = max(sparks, key=len)
                self.log(f"[News] 💡 Curiosity triggered for '{target}', but research limit reached. Adding as goal.")
                self.memory.add_goal(f"Research {target}", priority=3)
            else:
                if verbose: self.log("[News] Concepts appear familiar. Moving on.")
        
        # Cleanup after batch
        self.memory.prune_graph()
        # Check for consolidation at the end of the news session.
        if self.session_new_beliefs_count >= AlphaConfig.CONSOLIDATION_THRESHOLD:
            self.log(f"\n[System] Resting period: Consolidating after news session ({self.session_new_beliefs_count} new beliefs)...")
            self.consolidate(verbose=verbose)
            self.session_new_beliefs_count = 0 # Reset counter

        self.log("\n[News] Session complete.")

    def autonomous_discovery(self, verbose: bool = False, max_cycles: int = None):
        """
        The 'Inner Voice' loop. Fetches news, learns, and decides if it needs to research more.
        """
        print("\n--- 🧠 Starting Autonomous Discovery Loop (Press Ctrl+C to stop) ---")
        try:
            import utils.tools as tools # Lazy import to avoid circular dependencies if any
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
            if verbose: self.log(f"  [INVALID] Triple part is empty: ({s}, {p}, {o})")
            return False

        # Convert to string now that we know they are not None/empty/0, to prevent TypeErrors.
        s, p, o = str(s), str(p), str(o)

        # Check for excessive length (could be a sign of poor extraction)
        # Assumes AlphaConfig.MAX_TRIPLE_PART_LENGTH exists
        max_len = getattr(AlphaConfig, 'MAX_TRIPLE_PART_LENGTH', 100)
        if any(len(part) > max_len for part in [s, p, o]):
            if verbose: self.log(f"  [INVALID] Triple part too long: ({s}, {p}, {o})")
            return False

        # Check if subject or object are just stop words (after lowercasing)
        # Assumes AlphaConfig.STOP_WORDS exists
        stop_words = getattr(AlphaConfig, 'STOP_WORDS', set())
        if s.lower() in stop_words or o.lower() in stop_words:
            if verbose: self.log(f"  [INVALID] Subject or object is a stopword: ({s}, {p}, {o})")
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
            if verbose: self.log("[Curiosity] Suppressed: LLM budget exceeded.")
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
            if verbose: self.log(f"[Curiosity] Suppressed: System unstable ({stability:.2f}).")
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
        self.log(f"\n[Curiosity] Triggered! Interested in '{topic}' (Score: {belief['curiosity']:.2f})")
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
        tokens = self._correct_tokens(tokens)
        
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
            self.log(f"[System] Warning: Context retrieval failed (skipping): {e}")
            return []

    def close(self):
        self.memory.close()
