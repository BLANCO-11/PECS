import os
import threading
import time
import random
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from utils.core import PECSCore

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pecs_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

autonomous_thread = None
autonomous_active = False

# 1. DEFINE THE EVENT HANDLER
# This function will be called by PECSCore whenever a belief is created.
def core_event_handler(event_type, data):
    # It has access to socketio and can emit signals.
    socketio.emit(event_type, data)


# 2. UPDATE THE CORE CREATION
# Pass the handler into the core instance.
def create_core():
    return PECSCore(api_key, event_handler=core_event_handler)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/graph')
def get_graph():
    temp_core = create_core()

    try:
        # INCREASED LIMIT: Fetching more nodes to show full clusters
        # Fetching top 600 nodes by access time + randomness to get variety
        seeds = temp_core.memory.get_beliefs(sort_by='last_accessed')#, limit=5000)
        seed_ids = [b['id'] for b in seeds]

        if seed_ids:
            # Get the subgraph connecting these nodes
            nodes, edges = temp_core.memory.get_subgraph(seed_ids)
        else:
            nodes, edges = [], []

        # If graph is still too small, grab raw dump
        if len(nodes) < 50:
            nodes = temp_core.memory.get_beliefs(limit=4000)
            all_edges = temp_core.memory.get_edges()
            node_ids = set(n['id'] for n in nodes)
            edges = [e for e in all_edges if e['source_id'] in node_ids and e['target_id'] in node_ids]

        # Assign random initial positions to prevent the "0,0 explosion"
        for n in nodes:
            if 'x' not in n:
                n['x'] = random.uniform(-2000, 2000)
            if 'y' not in n:
                n['y'] = random.uniform(-2000, 2000)

        return jsonify({
            'nodes': nodes,
            'edges': edges
        })

    finally:
        temp_core.close()


@socketio.on('chat_message')
def handle_chat(data):
    user_input = data.get('message')
    deep_think = data.get('deep_think', False)
    verbose = data.get('verbose', False)

    if not user_input: return

    def process():
        core = create_core()
        try:
            # Result is now a dict: {'answer': str, 'reasoning': str, 'logs': list}
            result = core.process_interaction(user_input, verbose=verbose, deep_think=deep_think)
            
            # Normalize response if it's just a string (backward compatibility/error cases)
            if isinstance(result, str):
                socketio.emit('chat_response', {'response': result})
            else:
                socketio.emit('chat_response', {
                    'response': result.get('answer'),
                    'reasoning': result.get('reasoning'),
                    'logs': result.get('logs', [])
                })
                
        except Exception as e:
            print(f"Error in chat process: {e}")
            socketio.emit('chat_response', {'response': f"Error: {str(e)}"})
        finally:
            core.close()

    thread = threading.Thread(target=process)
    thread.start()

@socketio.on('toggle_autonomous')
def toggle_autonomous(data):
    global autonomous_active, autonomous_thread

    enable = data.get('enabled')

    if enable and not autonomous_active:
        autonomous_active = True

        def run_auto():
            print("[Server] Starting Autonomous Mode")
            core = create_core()
            try:
                while autonomous_active:
                    core.evaluate_curiosity_trigger(verbose=True)
                    next_goal = core.memory.get_next_goal()

                    if next_goal:
                        socketio.emit('status_update', {'status': f"Executing: {next_goal['description']}"})
                        if next_goal['description'].startswith("Research "):
                            topic = next_goal['description'].replace("Research ", "")
                            core.research_topic(topic, verbose=True, is_autonomous=True, plan_first=True)
                            core.memory.complete_goal(next_goal['id'])
                            # 3. REMOVED THE OLD SIGNAL FROM HERE
                        elif next_goal['description'].startswith("Execute Research "):
                            topic = next_goal['description'].replace("Execute Research ", "")
                            core.research_topic(topic, verbose=True, is_autonomous=True, plan_first=False)
                            core.memory.complete_goal(next_goal['id'])
                            # 3. REMOVED THE OLD SIGNAL FROM HERE
                        
                        time.sleep(2)
                        continue

                    socketio.emit('status_update', {'status': "Scanning Inputs..."})
                    time.sleep(5) # Idle wait

            finally:
                print("[Server] Autonomous Mode Stopped")
                core.close()

        autonomous_thread = threading.Thread(target=run_auto)
        autonomous_thread.start()

    elif not enable:
        autonomous_active = False
        if autonomous_thread:
            autonomous_thread.join(timeout=1)


if __name__ == '__main__':
    print("Starting PECS Server on http://localhost:5000")
    socketio.run(app, debug=True, port=5000)