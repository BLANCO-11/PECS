import re
import networkx as nx
from typing import List, Dict
from config import AlphaConfig
from storage import MemoryStore
from confidence import compute_confidence

class ActivationEngine:
    def __init__(self, store: MemoryStore):
        self.store = store

    def get_activated_beliefs(self, input_tokens: List[str]) -> List[Dict]:
        """
        1. Find seed nodes based on input tokens.
        2. Spread activation via NetworkX.
        3. Return top K beliefs.
        """
        
        # Pre-process tokens: strip punctuation and map aliases
        clean_tokens = []
        aliases = AlphaConfig.ALIASES
        stop_words = AlphaConfig.STOP_WORDS
        
        for t in input_tokens:
            clean = re.sub(r'[^\w\s]', '', t)
            if clean and clean not in stop_words:
                clean_tokens.append(clean)
                if clean in aliases:
                    clean_tokens.append(aliases[clean])
        
        if not clean_tokens:
            return []

        # 1. Optimized Seed Search (SQL-based)
        seed_beliefs = self.store.search_beliefs(clean_tokens)
        if not seed_beliefs:
            return []
            
        seed_ids = [b['id'] for b in seed_beliefs]
        
        # 2. Load Subgraph (Seeds + Neighbors)
        nodes, edges = self.store.get_subgraph(seed_ids)
        
        # Build Graph
        G = nx.DiGraph()
        belief_map = {b['id']: b for b in nodes}
        
        for b in nodes:
            G.add_node(b['id'], **b)
        for e in edges:
            G.add_edge(e['source_id'], e['target_id'], weight=e['weight'])

        # 3. Score Seeds
        initial_activation = {}
        for b_id, node in G.nodes(data=True):
            score = 0
            # Simple keyword matching
            content = f"{node['subject']} {node['predicate']} {node['object']}".lower()
            for token in clean_tokens:
                # Exact match preferred, partial match allowed
                if token in content.split():
                    score += 1.0
                elif token in content:
                    score += 0.5
            if score > 0:
                # Boost activation by evidence score so high-confidence facts win
                # even if they are isolated (low connectivity)
                initial_activation[b_id] = score * node.get('evidence_score', 1.0)

        # 4. Spread Activation (Simulated via Personal PageRank or BFS)
        # Using Personal PageRank as a proxy for spreading activation
        try:
            ppr = nx.pagerank(G, personalization=initial_activation, alpha=0.85)
        except:
            # Fallback if graph is disconnected or empty
            ppr = initial_activation

        # 5. Rank and Filter
        ranked_ids = sorted(ppr, key=ppr.get, reverse=True)[:AlphaConfig.TOP_K_ACTIVATION]
        
        activated_beliefs = []
        for rid in ranked_ids:
            b = belief_map[rid]
            # Compute dynamic confidence before returning
            b['confidence'] = compute_confidence(b)
            activated_beliefs.append(b)
            
        return activated_beliefs