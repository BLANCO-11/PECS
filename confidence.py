import time
import math
from typing import Dict
from config import AlphaConfig

def compute_confidence(belief: Dict) -> float:
    """
    Calculates dynamic confidence based on the formula:
    sigmoid(α*evidence - β*contradiction - γ*decay + δ*support)
    """
    now = time.time()
    temporal_decay = (now - belief['last_updated']) * belief['decay_rate']
    
    # Normalize decay to avoid massive negative numbers in long running systems
    temporal_decay = math.log(1 + temporal_decay) 

    val = (AlphaConfig.ALPHA * belief['evidence_score']) \
        - (AlphaConfig.BETA * belief['contradiction_score']) \
        - (AlphaConfig.GAMMA * temporal_decay) \
        + (AlphaConfig.DELTA * belief['structural_support_score'])
    
    return 1 / (1 + math.exp(-val))