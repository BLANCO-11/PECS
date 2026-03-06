import os
from dotenv import load_dotenv
from utils.core import PECSCore
def start_chat_interface(core: PECSCore):
    """Starts an interactive chat session with the PECS core."""
    print("\n--- PECS Chat Interface (Type 'exit' to quit) ---")
    print("Tip: Use /deep and/or /verbose flags before your query.")

    while True:
        try:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input:
                continue

            tokens = user_input.split()

            deep_think_override = False
            verbose_override = False
            query_tokens = []

            for token in tokens:
                t = token.lower()
                if t == "/deep":
                    deep_think_override = True
                elif t == "/verbose":
                    verbose_override = True
                else:
                    query_tokens.append(token)

            clean_input = " ".join(query_tokens)

            if not clean_input:
                continue

            response = core.process_interaction(
                clean_input,
                verbose=verbose_override,
                deep_think=deep_think_override
            )

            if response:
                print(f"\nPECS: {response}")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    load_dotenv()

    # Ensure you have GROQ_API_KEY in your env
    api_key = os.environ.get("GROQ_API_KEY")
    
    # Initialize core (set deep_think_mode=True if you want the deeper reasoning by default)
    core = PECSCore(api_key)
    
    # --- CHOOSE ONE MODE TO RUN ---

    # Mode 1: Interactive Chat
    start_chat_interface(core)
    

    # Mode 2: Pure Learning (Feeding Data) - No response generated
    # Good for bulk loading facts cheaply
    # core.learn("My dog's name is Shero and he was a golden labrador")
    # core.learn("Python is better than Java")
    
    # Mode 3: Batch Learn: Feed a "News Snippet" efficiently
    # news_text = "Global markets rallied today as tech stocks surged. The S&P 500 hit a record high. Meanwhile, gold prices dropped slightly due to a stronger dollar."
    # core.batch_learn(news_text)
    
    # Mode 4: Autonomous Discovery (Self-Learning)
    # The system will fetch news, learn headlines, and research unknown topics automatically.
    # core.autonomous_discovery(verbose=True)
    
    # Mode 5: Focused Research (Self-Learning but sticks to a topic)
    # core.research_topic("research ronaldo", verbose=True)
    
    # You can then query what it learned after a learning session:
    # Use deep_think=True for a more comprehensive, synthesized answer at the cost of more tokens.
    # response = core.process_interaction("Tell me about rajasthan", verbose=True)
    # print(response)
    
    # core.read_news()

    # This is basically sleep equivalent for this sytem, cleans up neurons/nodes that are not relevant
    # core.consolidate()
    

    core.close()
