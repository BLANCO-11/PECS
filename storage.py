import sqlite3
import uuid
import time
from config import AlphaConfig

class MemoryStore:
    def __init__(self, db_path="alpha_core.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        
        # Beliefs Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS beliefs (
                id TEXT PRIMARY KEY,
                type TEXT,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                evidence_score REAL DEFAULT 1.0,
                contradiction_score REAL DEFAULT 0.0,
                structural_support_score REAL DEFAULT 0.0,
                decay_rate REAL DEFAULT 0.1,
                last_updated REAL,
                created_at REAL,
                usage_count INTEGER DEFAULT 0,
                UNIQUE(subject, predicate, object)
            )
        """)
        
        # Edges Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT,
                target_id TEXT,
                type TEXT,
                weight REAL,
                last_activated REAL,
                FOREIGN KEY(source_id) REFERENCES beliefs(id),
                FOREIGN KEY(target_id) REFERENCES beliefs(id)
            )
        """)
        
        # Performance Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_object ON beliefs(object)")
        
        # Goals Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id TEXT PRIMARY KEY,
                description TEXT,
                priority INTEGER,
                status TEXT
            )
        """)

        # Migration: Add created_at if missing (for existing databases)
        try:
            cur.execute("SELECT created_at FROM beliefs LIMIT 1")
        except sqlite3.OperationalError:
            print("[System] Migrating DB: Adding 'created_at' to beliefs table...")
            cur.execute("ALTER TABLE beliefs ADD COLUMN created_at REAL")
            cur.execute("UPDATE beliefs SET created_at = last_updated")
        
        self.conn.commit()

    def add_belief(self, subj, pred, obj, b_type="fact", weight=1.0):
        """Upsert logic: Update evidence if exists, Insert if new."""
        cur = self.conn.cursor()
        now = time.time()
        
        b_id = None
        is_new = False

        # Check existence
        cur.execute("SELECT id, evidence_score FROM beliefs WHERE subject=? AND predicate=? AND object=?", 
                    (subj, pred, obj))
        row = cur.fetchone()
        
        if row:
            # Update existing
            b_id = row['id']
            new_score = row['evidence_score'] + weight
            cur.execute("""
                UPDATE beliefs 
                SET evidence_score = ?, last_updated = ?, usage_count = usage_count + 1 
                WHERE id = ?
            """, (new_score, now, b_id))
            is_new = False
        else:
            # Insert new
            b_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO beliefs (id, type, subject, predicate, object, last_updated, evidence_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (b_id, b_type, subj, pred, obj, now, weight, now))
            is_new = True

        # Contradiction Detection
        self._check_contradictions(b_id, subj, pred, obj)

        return b_id, is_new

    def _check_contradictions(self, b_id, subj, pred, obj):
        """Detects simple contradictions and adds edges."""
        cur = self.conn.cursor()

        def add_conflict(id1, id2):
            # Check if connection already exists to prevent score inflation
            cur.execute("SELECT 1 FROM edges WHERE source_id=? AND target_id=? AND type='contradicts'", (id1, id2))
            if cur.fetchone():
                return # Already marked
            
            self.add_edge(id1, id2, 'contradicts', weight=1.0)
            self.add_edge(id2, id1, 'contradicts', weight=1.0)
            self._increment_contradiction(id1)
            self._increment_contradiction(id2)
        
        # 1. Inverse Predicates
        inverse = AlphaConfig.INVERSE_PREDICATES.get(pred)
        if inverse:
            cur.execute("SELECT id FROM beliefs WHERE subject=? AND predicate=? AND object=?", (subj, inverse, obj))
            rows = cur.fetchall()
            for row in rows:
                add_conflict(b_id, row['id'])

        # 2. Symmetric Contradictions (preferred_over)
        if pred == "preferred_over":
            cur.execute("SELECT id FROM beliefs WHERE subject=? AND predicate=? AND object=?", (obj, pred, subj))
            rows = cur.fetchall()
            for row in rows:
                add_conflict(b_id, row['id'])

    def _increment_contradiction(self, b_id):
        cur = self.conn.cursor()
        cur.execute("UPDATE beliefs SET contradiction_score = contradiction_score + 1 WHERE id=?", (b_id,))

    def add_edge(self, source_id, target_id, edge_type, weight=0.5):
        """Saves a relationship between two beliefs to the database."""
        cur = self.conn.cursor()
        now = time.time()
        edge_id = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO edges (id, source_id, target_id, type, weight, last_activated)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING
        """, (edge_id, source_id, target_id, edge_type, weight, now))

    def is_research_goal_achieved(self, topic):
        """Checks if a research goal for this topic has already been completed."""
        cur = self.conn.cursor()
        pattern_exact = f"Research {topic}"
        pattern_sub = f"Research %: {topic}"
        cur.execute("SELECT 1 FROM goals WHERE status = 'achieved' AND (description = ? OR description LIKE ?)", (pattern_exact, pattern_sub))
        return cur.fetchone() is not None

    def add_goal(self, description, priority=1):
        """Adds a new goal if it doesn't already exist in pending state."""
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM goals WHERE description = ? AND status = 'pending'", (description,))
        row = cur.fetchone()
        if row:
            return row['id']
            
        g_id = str(uuid.uuid4())
        cur.execute("INSERT INTO goals (id, description, priority, status) VALUES (?, ?, ?, ?)",
                    (g_id, description, priority, 'pending'))
        self.conn.commit()
        return g_id

    def get_next_goal(self):
        """Retrieves the highest priority pending goal."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM goals WHERE status = 'pending' ORDER BY priority DESC LIMIT 1")
        row = cur.fetchone()
        return dict(row) if row else None

    def complete_goal(self, goal_id):
        """Marks a goal as achieved."""
        cur = self.conn.cursor()
        cur.execute("UPDATE goals SET status = 'achieved' WHERE id = ?", (goal_id,))
        self.conn.commit()

    def search_beliefs(self, tokens: list):
        """Finds beliefs containing any of the tokens (SQL optimized)."""
        if not tokens: return []
        # Simple LIKE query for each token against subject/predicate/object
        # Note: In production, FTS5 (Full Text Search) would be better.
        conditions = []
        params = []
        for t in tokens:
            conditions.append("(subject LIKE ? OR predicate LIKE ? OR object LIKE ?)")
            p = f"%{t}%"
            params.extend([p, p, p])
        
        # Optimization: Prioritize high-evidence beliefs so they aren't cut off by LIMIT
        # Sort by Evidence -> Support (Connectivity) -> Usage. This ensures definitions bubbles up.
        query = f"SELECT * FROM beliefs WHERE {' OR '.join(conditions)} ORDER BY evidence_score DESC, structural_support_score DESC, usage_count DESC LIMIT 20"
        cur = self.conn.cursor()
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]

    def get_subgraph(self, seed_ids: list):
        """Fetches the 1-hop neighborhood for a set of belief IDs."""
        if not seed_ids: return [], []
        
        placeholders = ','.join('?' for _ in seed_ids)
        
        # 1. Get edges connected to seeds
        sql_edges = f"SELECT * FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})"
        cur = self.conn.cursor()
        cur.execute(sql_edges, seed_ids + seed_ids)
        edges = [dict(row) for row in cur.fetchall()]
        
        # 2. Get all involved nodes (seeds + neighbors)
        node_ids = set(seed_ids)
        for e in edges:
            node_ids.add(e['source_id'])
            node_ids.add(e['target_id'])
            
        if not node_ids: return [], []
        
        placeholders_nodes = ','.join('?' for _ in node_ids)
        sql_nodes = f"SELECT * FROM beliefs WHERE id IN ({placeholders_nodes})"
        cur.execute(sql_nodes, list(node_ids))
        nodes = [dict(row) for row in cur.fetchall()]
        
        return nodes, edges

    def merge_beliefs(self, keep_id, merge_id):
        """Merges merge_id INTO keep_id and deletes merge_id."""
        if keep_id == merge_id: return
        
        cur = self.conn.cursor()
        # Re-point edges
        cur.execute("UPDATE edges SET source_id = ? WHERE source_id = ?", (keep_id, merge_id))
        cur.execute("UPDATE edges SET target_id = ? WHERE target_id = ?", (keep_id, merge_id))
        
        # Transfer usage stats
        cur.execute("SELECT usage_count, evidence_score FROM beliefs WHERE id = ?", (merge_id,))
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE beliefs SET usage_count = usage_count + ?, evidence_score = evidence_score + ? WHERE id = ?", 
                        (row['usage_count'], row['evidence_score'], keep_id))
            
        # Delete the old node
        cur.execute("DELETE FROM beliefs WHERE id = ?", (merge_id,))
        self.conn.commit()

    def recompute_structural_support(self):
        """Updates structural_support_score for all beliefs based on connectivity."""
        cur = self.conn.cursor()
        
        # Fetch all edges to build graph in memory
        cur.execute("SELECT source_id, target_id, type FROM edges")
        edges = cur.fetchall()
        
        # Calculate in-degrees (weighted by usage/existence)
        in_degrees = {}
        for e in edges:
            if e['type'] == 'contradicts': continue
            target = e['target_id']
            in_degrees[target] = in_degrees.get(target, 0) + 1
            
        # Reset and Update
        cur.execute("UPDATE beliefs SET structural_support_score = 0")
        if in_degrees:
            updates = [(score, b_id) for b_id, score in in_degrees.items()]
            cur.executemany("UPDATE beliefs SET structural_support_score = ? WHERE id = ?", updates)
        self.conn.commit()

    def get_all_beliefs(self):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM beliefs")
        return [dict(row) for row in cur.fetchall()]

    def get_edges(self):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM edges")
        return [dict(row) for row in cur.fetchall()]

    def forget_weak_beliefs(self):
        """
        Removes beliefs that have decayed below a useful threshold.
        Simulates long-term forgetting of trivial/unused facts.
        """
        cur = self.conn.cursor()
        # Policy: Forget facts that have default evidence (1.0), 
        # haven't been used much (< 3 times), and are older than retention period.
        cutoff_time = time.time() - (AlphaConfig.FORGET_RETENTION_DAYS * 24 * 60 * 60)
        
        cur.execute("""
            DELETE FROM beliefs 
            WHERE evidence_score <= 1.0 
            AND usage_count < 3 
            AND last_updated < ?
        """, (cutoff_time,))
        
        # Also forget highly contradicted beliefs regardless of age
        cur.execute("""
            DELETE FROM beliefs 
            WHERE contradiction_score > evidence_score
        """)
        self.conn.commit()

    def prune_graph(self):
        """Removes lowest stability beliefs if over limit."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) as count FROM beliefs")
        count = cur.fetchone()['count']
        
        if count > AlphaConfig.MAX_BELIEFS:
            # Optimized Pruning:
            # Prioritize removing contradicted beliefs (low net score), then low usage, then old.
            # Sort ASC so the "worst" beliefs are at the top to be deleted.
            cur.execute("""
                DELETE FROM beliefs 
                WHERE id IN (
                    SELECT id FROM beliefs 
                    ORDER BY (evidence_score - contradiction_score) ASC, usage_count ASC, last_updated ASC 
                    LIMIT ?
                )
            """, (count - AlphaConfig.MAX_BELIEFS,))
            self.conn.commit()

    def close(self):
        self.conn.close()