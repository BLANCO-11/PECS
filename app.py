import os
from core import AlphaCognitiveCore

# --- Usage Example ---

if __name__ == "__main__":
    # Ensure you have GROQ_API_KEY in your env
    api_key = "REMOVED"#os.environ.get("GROQ_API_KEY", "gsk_placeholder_key")
    
    core = AlphaCognitiveCore(api_key)
    
    # 1. Pure Learning (Feeding Data) - No response generated
    # Good for bulk loading facts cheaply
    # core.learn("My dog's name is Shero and he was a golden labrador")
    # core.learn("Python is better than Java")
    
    # Batch Learn: Feed a "News Snippet" efficiently
    # news_text = "Global markets rallied today as tech stocks surged. The S&P 500 hit a record high. Meanwhile, gold prices dropped slightly due to a stronger dollar."
    # core.batch_learn(news_text)
    
    # 5. Autonomous Discovery (Self-Learning)
    # The system will fetch news, learn headlines, and research unknown topics automatically.
    # core.autonomous_discovery(verbose=True)
    
    # You can then query what it learned:
    core.process_interaction("what do you know about plan crash in bolivia?", verbose=True)
    # core.consolidate()
    
    core.close()
