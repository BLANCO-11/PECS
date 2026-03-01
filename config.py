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
    TOP_K_ACTIVATION = 5
    MIN_CONFIDENCE_FOR_CONTEXT = 0.6
    FORGET_RETENTION_DAYS = 3  # Days to keep low-evidence, unused beliefs
    CONSOLIDATION_THRESHOLD = 10  # Trigger consolidation after learning this many new beliefs

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
        'will', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves'
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