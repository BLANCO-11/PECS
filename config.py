class AlphaConfig:
    # Identity Parameters (The "Soul" of the system)
    MAX_BELIEFS = 100000
    MAX_EDGES = 400000
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
    # TOP_K_SYNTHESIZE = 10
    MIN_CONFIDENCE_FOR_CONTEXT = 0.6
    MIN_LLM_CONFIDENCE_TO_INGEST = 0.75 # New: Threshold for initial LLM extraction
    FORGET_RETENTION_DAYS = 3  # Days to keep low-evidence, unused beliefs
    CONSOLIDATION_THRESHOLD = 50  # Trigger consolidation after learning this many new beliefs
    MEMORY_PRESSURE_THRESHOLD = 0.8 # Only decay/prune when memory is > 80% full

    # Curiosity & Stability
    CURIOSITY_THRESHOLD = 0.5
    STABILITY_THRESHOLD = 0.3
    LLM_BUDGET_WINDOW = 10
    MAX_LLM_CALLS_IN_WINDOW = 20
    # Research Time & Resource Limits
    RESEARCH_MAX_TIME_SECONDS = 60  # Default max duration for a single research_topic call
    RESEARCH_MAX_TIME_DEEPTHINK_SECONDS = 180 # Max duration for a deep think research call
    RESEARCH_IGNORE_TIME_LIMIT = False # Override flag to ignore time limits for debugging/special cases
    RESEARCH_MAX_SOURCES = 3 # Default number of web sources to process
    RESEARCH_MAX_SOURCES_DEEPTHINK = 5 # Number of web sources to process in deep think mode
    MAX_CURIOSITY_GOALS_WINDOW = 2
    CURIOSITY_COOLDOWN_SECONDS = 60
    MAX_NEWS_HEADLINES = 5
    MAX_CHUNK_SIZE = 4000
    
    INVERSE_PREDICATES = {
        "is_a": "not_is_a",
        "caused_by": "prevents",
        "has": "not_has",
        "likes": "dislikes"
    }
    # LLM Models
    REASONING_MODEL_FAST = "llama-3.1-8b-instant"
    REASONING_MODEL_DEEP = "llama-3.1-8b-instant"
    EXTRACTION_MODEL = "llama-3.1-8b-instant"
    SYNTHESIZE_MODEL = "llama-3.1-8b-instant"
    MERGE_MODEL = "llama-3.1-8b-instant"
    RESEARCH_PLANNER_MODEL = "llama-3.1-8b-instant"
    RELEVANCE_CHECK_MODEL = "llama-3.1-8b-instant"

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