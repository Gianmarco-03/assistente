import json

SIMULATION_FILE = "simulation.json"

def load_state():
    # Apri il file (ad esempio 'dati.json')
    with open(SIMULATION_FILE, "r", encoding="utf-8") as f:
       return  json.load(f)
    
def save_state(new_state):
    with open(SIMULATION_FILE, "w", encoding="utf-8") as f:
        json.dump(new_state, f, indent=4)
