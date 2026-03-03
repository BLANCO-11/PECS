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
                status TEXT,
                parent_id TEXT,
                FOREIGN KEY(parent_id) REFERENCES goals(id)
            )
        """)

        # Migration: Add created_at if missing (for existing databases)
        try:
            cur.execute("SELECT created_at FROM beliefs LIMIT 1")
        except sqlite3.OperationalError:
            print("[System] Migrating DB: Adding 'created_at' to beliefs table...")
            cur.execute("ALTER TABLE beliefs ADD COLUMN created_at REAL")
            cur.execute("UPDATE beliefs SET created_at = last_updated")

        # Migration: Add parent_id to goals if missing
        try:
            cur.execute("SELECT parent_id FROM goals LIMIT 1")
        except sqlite3.OperationalError:
            print("[System] Migrating DB: Adding 'parent_id' to goals table...")
            cur.execute("ALTER TABLE goals ADD COLUMN parent_id TEXT")
        
        # FTS5 Virtual Table for fast text search
        # We store the UUID 'id' as an unindexed column to map back to the main table
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS beliefs_fts 
            USING fts5(id UNINDEXED, subject, predicate, object)
        """)

        # Sync check: If FTS is empty but beliefs exist, populate it (Migration)
        self._sync_fts_if_needed()
        
        self.conn.commit()

    def _sync_fts_if_needed(self):
        cur = self.conn.cursor()
        cur.execute("SELECT count(*) FROM beliefs_fts")
        if cur.fetchone()[0] == 0:
            cur.execute("SELECT count(*) FROM beliefs")
            if cur.fetchone()[0] > 0:
                print("[System] Migrating knowledge to FTS5 index...")
                cur.execute("INSERT INTO beliefs_fts (id, subject, predicate, object) SELECT id, subject, predicate, object FROM beliefs")

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
            
            # Update FTS Index
            cur.execute("INSERT INTO beliefs_fts (id, subject, predicate, object) VALUES (?, ?, ?, ?)", (b_id, subj, pred, obj))
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

    def add_goal(self, description, priority=1, parent_id=None):
        """Adds a new goal if it doesn't already exist in pending state."""
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM goals WHERE description = ? AND status = 'pending'", (description,))
        row = cur.fetchone()
        if row:
            return row['id']
            
        g_id = str(uuid.uuid4())
        cur.execute("INSERT INTO goals (id, description, priority, status, parent_id) VALUES (?, ?, ?, ?, ?)",
                    (g_id, description, priority, 'pending', parent_id))
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

    def search_beliefs(self, tokens: list, limit: int = 50):
        """Finds beliefs containing any of the tokens (SQL optimized)."""
        if not tokens: return []
        
        # FTS5 Search: Much faster O(log N) compared to LIKE O(N)
        # Construct query: "token1 OR token2 OR token3"
        fts_query = " OR ".join(f'"{t}"' for t in tokens if t.isalnum())
        if not fts_query: return []

        cur = self.conn.cursor()
        # We join with the main table to get the full belief data and sort by quality
        sql = f"""
            SELECT b.* 
            FROM beliefs_fts f 
            JOIN beliefs b ON f.id = b.id 
            WHERE beliefs_fts MATCH ? 
            ORDER BY b.evidence_score DESC, b.structural_support_score DESC 
            LIMIT ?
        """
        cur.execute(sql, (fts_query, limit))
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
        # Delete from FTS
        cur.execute("DELETE FROM beliefs_fts WHERE id = ?", (merge_id,))
        
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

    def get_beliefs(self, sort_by='last_accessed', limit=100):
        """Retrieves a subset of beliefs sorted by a specific criteria."""
        cur = self.conn.cursor()
        if sort_by == 'last_accessed':
            order = "last_updated DESC"
        elif sort_by == 'evidence':
            order = "evidence_score DESC"
        else:
            order = "last_updated DESC"
            
        cur.execute(f"SELECT * FROM beliefs ORDER BY {order} LIMIT ?", (limit,))
        return [dict(row) for row in cur.fetchall()]

    def apply_decay(self):
        """Decays the evidence score of all beliefs."""
        factor = 1.0 - AlphaConfig.GAMMA
        cur = self.conn.cursor()
        cur.execute("UPDATE beliefs SET evidence_score = evidence_score * ?", (factor,))
        self.conn.commit()

    def get_edges(self):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM edges")
        return [dict(row) for row in cur.fetchall()]

    def forget_weak_beliefs(self):
        """
        Removes beliefs that have decayed below a useful threshold or are highly contradicted.
        Simulates long-term forgetting of trivial/unused facts and their associated edges.
        """
        cur = self.conn.cursor()
        cutoff_time = time.time() - (AlphaConfig.FORGET_RETENTION_DAYS * 24 * 60 * 60)
        
        # Get IDs of all beliefs to be deleted from both conditions
        cur.execute("""
            SELECT id FROM beliefs
            WHERE (evidence_score <= 1.0 AND usage_count < 3 AND last_updated < ?)
            OR (contradiction_score > evidence_score)
        """, (cutoff_time,))
        ids_to_delete = [row['id'] for row in cur.fetchall()]
        
        if not ids_to_delete:
            return

        # Batch deletes to avoid "too many SQL variables" error.
        # Using 450 because the edge deletion uses the list twice. 450*2=900, safely under the 999 limit.
        batch_size = 450
        for i in range(0, len(ids_to_delete), batch_size):
            batch = ids_to_delete[i:i + batch_size]
            placeholders = ','.join('?' for _ in batch)

            # 1. Delete associated edges pointing to or from the deleted beliefs
            cur.execute(f"DELETE FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})", batch + batch)

            # 2. Delete from FTS table
            cur.execute(f"DELETE FROM beliefs_fts WHERE id IN ({placeholders})", batch)

            # 3. Delete the beliefs themselves
            cur.execute(f"DELETE FROM beliefs WHERE id IN ({placeholders})", batch)

        self.conn.commit()

    def prune_graph(self):
        """Removes lowest stability beliefs and their edges if over limit."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) as count FROM beliefs")
        count = cur.fetchone()['count']
        
        if count > AlphaConfig.MAX_BELIEFS:
            limit = count - AlphaConfig.MAX_BELIEFS
            cur.execute(f"SELECT id FROM beliefs ORDER BY (evidence_score - contradiction_score) ASC, usage_count ASC, last_updated ASC LIMIT {limit}")
            ids_to_delete = [row['id'] for row in cur.fetchall()]

            if ids_to_delete:
                # Batch deletes to avoid "too many SQL variables" error.
                batch_size = 450 # Using 450 because we use the list twice for edges.
                for i in range(0, len(ids_to_delete), batch_size):
                    batch = ids_to_delete[i:i + batch_size]
                    placeholders = ','.join('?' for _ in batch)
                    # 1. Delete associated edges
                    cur.execute(f"DELETE FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})", batch + batch)
                    # 2. Delete from FTS
                    cur.execute(f"DELETE FROM beliefs_fts WHERE id IN ({placeholders})", batch)
                    # 3. Delete beliefs
                    cur.execute(f"DELETE FROM beliefs WHERE id IN ({placeholders})", batch)
                
            self.conn.commit()

    def clean_orphaned_edges(self):
        """
        Finds and removes edges that point to non-existent beliefs.
        This is a data integrity maintenance function to clean up corruption
        from previous versions or unclean shutdowns.
        """
        cur = self.conn.cursor()
        
        # 1. Get all valid belief IDs into a fast lookup set
        cur.execute("SELECT id FROM beliefs")
        valid_belief_ids = {row['id'] for row in cur.fetchall()}
        
        if not valid_belief_ids:
            # If there are no beliefs, there should be no edges.
            cur.execute("DELETE FROM edges")
            self.conn.commit()
            return

        # 2. Find all edge IDs that are invalid
        cur.execute("SELECT id, source_id, target_id FROM edges")
        orphaned_edge_ids = [
            edge['id'] for edge in cur.fetchall() 
            if edge['source_id'] not in valid_belief_ids or edge['target_id'] not in valid_belief_ids
        ]
        
        if orphaned_edge_ids:
            print(f"[System] Data Integrity: Found and removed {len(orphaned_edge_ids)} orphaned edge(s).")
            # Batch delete to avoid "too many SQL variables" error
            batch_size = 900 # SQLite's default variable limit is 999
            for i in range(0, len(orphaned_edge_ids), batch_size):
                batch = orphaned_edge_ids[i:i + batch_size]
                placeholders = ','.join('?' for _ in batch)
                cur.execute(f"DELETE FROM edges WHERE id IN ({placeholders})", batch)
            self.conn.commit()

    def close(self):
        self.conn.close()