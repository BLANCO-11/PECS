import os
from dotenv import load_dotenv
from core import PECSCore

if __name__ == "__main__":
    load_dotenv()

    # Ensure you have GROQ_API_KEY in your env
    api_key = os.environ.get("GROQ_API_KEY")
    
    # Initialize core (set deep_think_mode=True if you want the deeper reasoning by default)
    core = PECSCore(api_key)
    
    #  Pure Learning (Feeding Data) - No response generated
    # Good for bulk loading facts cheaply
    # core.learn("My dog's name is Shero and he was a golden labrador")
    # core.learn("Python is better than Java")
    
    # Batch Learn: Feed a "News Snippet" efficiently
    # news_text = "Global markets rallied today as tech stocks surged. The S&P 500 hit a record high. Meanwhile, gold prices dropped slightly due to a stronger dollar."
    # core.batch_learn(news_text)
    
    #  Autonomous Discovery (Self-Learning)
    # The system will fetch news, learn headlines, and research unknown topics automatically.
    # core.autonomous_discovery(verbose=True)
    
    #  Autonomous Discovery (Self-Learning but sticks to a topic)
    # core.research_topic("Ancient India", verbose=True)
    
    # You can then query what it learned:
    core.process_interaction("Tell me about buddhism in India", verbose=True)

    # This is basically sleep equivalent for this sytem, cleans up neurons/nodes that are not relevant
    # core.consolidate()
    
    core.close()
