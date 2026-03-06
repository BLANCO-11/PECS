import re
import math
import networkx as nx
from typing import List, Dict
from utils.config import AlphaConfig
from utils.confidence import compute_confidence


class ActivationEngine:

    def __init__(self, store):
        self.store = store

    def _tokenize(self, text):
        return set(re.findall(r'\w+', text.lower()))

    def get_activated_beliefs(self, input_tokens: List[str], limit=AlphaConfig.TOP_K_ACTIVATION):

        stop_words = AlphaConfig.STOP_WORDS
        aliases = AlphaConfig.ALIASES

        clean_tokens = []

        for t in input_tokens:

            clean = re.sub(r"[^\w\s]", "", t.lower())

            if not clean or clean in stop_words:
                continue

            clean_tokens.append(clean)

            if clean in aliases:
                clean_tokens.append(aliases[clean])

        if not clean_tokens:
            return []

        seed_beliefs = self.store.search_beliefs(clean_tokens)

        if not seed_beliefs:
            return []

        seed_ids = [b["id"] for b in seed_beliefs]

        depth = 2 if limit > AlphaConfig.TOP_K_ACTIVATION else 1

        nodes, edges = self.store.get_subgraph(seed_ids, depth=depth)

        if len(nodes) > AlphaConfig.MAX_GRAPH_SIZE:
            nodes = nodes[:AlphaConfig.MAX_GRAPH_SIZE]

        belief_map = {b["id"]: b for b in nodes}

        G = nx.DiGraph()

        for b in nodes:
            G.add_node(b["id"], **b)

        edge_count = 0

        for e in edges:

            if e["weight"] < 0.3:
                continue

            G.add_edge(e["source_id"], e["target_id"], weight=e["weight"])

            edge_count += 1

            if edge_count > AlphaConfig.MAX_GRAPH_EDGES:
                break

        initial_activation = {}

        for b_id, node in G.nodes(data=True):

            if "subject" not in node:
                continue

            content = f"{node['subject']} {node['predicate']} {node['object']}".lower()

            tokens = self._tokenize(content)

            score = 0

            for t in clean_tokens:

                if t in tokens:
                    score += 1.0

                elif t in content:
                    score += 0.5

            if score > 0:

                evidence = node.get("evidence_score", 1.0)
                support = node.get("structural_support_score", 0)

                # Activation Normalization: Logarithmic scaling to prevent energy explosion
                norm_evidence = math.log(1 + evidence)
                norm_support = math.log(1 + support)

                initial_activation[b_id] = score * (1.0 + 0.2 * norm_evidence + 0.1 * norm_support)

        if not initial_activation:
            return []

        try:

            ppr = nx.pagerank(
                G,
                personalization=initial_activation,
                alpha=0.85,
                max_iter=100
            )

        except Exception:
            ppr = initial_activation

        # Normalize output scores relative to the best match (Max-Scaling)
        # This ensures scores are 0.0-1.0 regardless of graph size
        max_activation = max(ppr.values()) if ppr else 1.0

        ranked_ids = sorted(ppr, key=ppr.get, reverse=True)[:limit]

        activated = []

        for rid in ranked_ids:

            b = belief_map.get(rid)

            if not b:
                continue

            content = f"{b['subject']} {b['predicate']} {b['object']}".lower()

            if not any(t in content for t in clean_tokens):
                continue

            b["confidence"] = compute_confidence(b)
            b["relevance_score"] = ppr.get(rid, 0) / max_activation if max_activation > 0 else 0

            activated.append(b)

        return activated