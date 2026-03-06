class AlphaConfig:

    # Identity Parameters
    MAX_BELIEFS = 100000
    MAX_EDGES = 400000
    MAX_DEPTH = 5
    CONTRADICTION_THRESHOLD = 0.6

    # Confidence Hyperparameters
    ALPHA = 1.0
    BETA = 1.2
    GAMMA = 0.05
    DELTA = 0.5

    # Activation limits
    TOP_K_ACTIVATION = 8
    TOP_K_DEEP_THINK = 18

    # Reasoning limits
    FACT_SELECTOR_LIMIT = 30
    MAX_REASON_FACTS = 25
    MIN_REASON_FACTS = 10
    
    # Graph limits
    MAX_GRAPH_SIZE = 500
    MAX_GRAPH_EDGES = 1000

    MIN_CONFIDENCE_FOR_CONTEXT = 0.6
    MIN_LLM_CONFIDENCE_TO_INGEST = 0.75

    # Memory maintenance
    FORGET_RETENTION_DAYS = 3
    CONSOLIDATION_THRESHOLD = 50
    MEMORY_PRESSURE_THRESHOLD = 0.8

    # Curiosity
    CURIOSITY_THRESHOLD = 0.65
    STABILITY_THRESHOLD = 0.4
    LLM_BUDGET_WINDOW = 10
    MAX_LLM_CALLS_IN_WINDOW = 20
    MAX_CURIOSITY_GOALS_WINDOW = 2
    CURIOSITY_COOLDOWN_SECONDS = 60

    # Research limits
    RESEARCH_MAX_TIME_SECONDS = 30
    RESEARCH_MAX_TIME_DEEPTHINK_SECONDS = 120
    RESEARCH_IGNORE_TIME_LIMIT = False
    RESEARCH_MAX_SOURCES = 3
    RESEARCH_MAX_SOURCES_DEEPTHINK = 6

    MAX_NEWS_HEADLINES = 5
    MAX_CHUNK_SIZE = 4000

    # Debug
    DEBUG_TOKENS = True
    DEBUG_REASONING = False

    # Reasoning triggers
    REASONING_TRIGGERS = {
        "why","how","cause","reason","effect",
        "relationship","related","connect","link",
        "difference","compare"
    }

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

    # Stop words
    STOP_WORDS = {
        'a','about','above','after','again','against','all','am','an','and',
        'any','are','as','at','be','because','been','before','being',
        'below','between','both','but','by','can','did','do','does',
        'doing','down','during','each','few','for','from','further','had',
        'has','have','having','he','her','here','hers','herself','him',
        'himself','his','how','if','in','into','is','it','its','itself',
        'just','me','more','most','my','myself','no','nor','not','now',
        'of','off','on','once','only','or','other','our','ours',
        'ourselves','out','over','own','same','she','should','so',
        'some','such','than','that','the','their','theirs','them',
        'themselves','then','there','these','they','this','those',
        'through','to','too','under','until','up','very','was','we',
        'were','what','when','where','which','while','who','whom','why',
        'will','with','you','your','yours','yourself','yourselves',
        'tell','describe','explain','show','find','search',
        'information','thing','things','something','someone','anything'
    }

    COMMON_ENTITIES = {
        'league','union','united','association','organization','group',
        'party','club','agency','council','committee','department',
        'ministry','system','network','service','force','army'
    }

    ALIASES = {
        "i":"user",
        "my":"user",
        "me":"user",
        "myself":"user",
        "we":"user"
    }