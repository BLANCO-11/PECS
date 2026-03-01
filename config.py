class AlphaConfig:
    # Identity Parameters (The "Soul" of the system)
    MAX_BELIEFS = 5000
    MAX_EDGES = 20000
    MAX_DEPTH = 5
    CONTRADICTION_THRESHOLD = 0.6
    
    # Confidence Hyperparameters
    ALPHA = 1.0  # Evidence weight
    BETA = 1.2   # Contradiction penalty
    GAMMA = 0.05 # Decay rate factor
    DELTA = 0.5  # Structural support weight
    
    # System Limits
    MAX_LLM_CALLS = 2
    ACTIVATION_DECAY = 0.5
    TOP_K_ACTIVATION = 10
    TOP_K_DEEP_THINK = 25
    TOP_K_SYNTHESIZE = 10
    MIN_CONFIDENCE_FOR_CONTEXT = 0.6
    FORGET_RETENTION_DAYS = 3  # Days to keep low-evidence, unused beliefs
    CONSOLIDATION_THRESHOLD = 15  # Trigger consolidation after learning this many new beliefs

    # Curiosity & Stability
    CURIOSITY_THRESHOLD = 0.5
    STABILITY_THRESHOLD = 0.3
    LLM_BUDGET_WINDOW = 10
    MAX_LLM_CALLS_IN_WINDOW = 20
    RESEARCH_MAX_TIME_SECONDS = 60  # Max duration for a single research_topic call (prevents infinite rabbit holes)
    MAX_CURIOSITY_GOALS_WINDOW = 2
    CURIOSITY_COOLDOWN_SECONDS = 300
    
    INVERSE_PREDICATES = {
        "is_a": "not_is_a",
        "caused_by": "prevents",
        "has": "not_has",
        "likes": "dislikes"
    }

    # A set of common English words to ignore during belief retrieval to reduce noise.
    STOP_WORDS = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
        'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
        'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do', 'does',
        'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had',
        'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him',
        'himself', 'his', 'how', 'if', 'in', 'into', 'is', 'it', 'its', 'itself',
        'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now',
        'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours',
        'ourselves', 'out', 'over', 'own', 's', 'same', 'she', 'should', 'so',
        'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them',
        'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
        'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we',
        'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
        'will', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves',
        'tell', 'describe', 'explain', 'show', 'find', 'search',
        'may', 'might', 'could', 'can', 'would', 'should', 'must',
        'one', 'two', 'three', 'first', 'second', 'third',
        'result', 'occur', 'caused', 'using', 'use', 'used', 'made', 'make'
    }

    # Common entity suffixes/types that cause noise if matched in isolation
    # e.g. "Hanseatic League" matching "Champions League" just on "League"
    COMMON_ENTITIES = {
        'league', 'union', 'united', 'association', 'organization', 'group', 
        'party', 'club', 'agency', 'council', 'committee', 'department', 
        'ministry', 'system', 'network', 'service', 'force', 'army'
    }

    # Natural Language Configuration
    ALIASES = {
        "i": "user", 
        "my": "user", 
        "me": "user", 
        "myself": "user",
        "we": "user"
    }
    
    RESPONSE_INTROS = [
        "I recall that", 
        "My memory indicates that", 
        "Based on what I've learned,", 
        "It appears that", 
        "I believe",
        "Records show that"
    ]
    
    RESPONSE_TRANSITIONS = [
        "Also,", 
        "Additionally,", 
        "Furthermore,", 
        "It is also worth noting that",
        "Moreover,"
    ]
    
    # Map raw predicates to natural language for deterministic responses
    PREDICATE_MAP = {
        "preferred_over": "is preferred over",
        "is_a": "is a",
        "caused_by": "was caused by",
        "due_to": "is due to",
        "related_to": "is related to",
        "prefers": "prefers",
        "has": "has",
        "part_of": "is part of"
    }