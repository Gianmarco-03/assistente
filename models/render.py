from joblib import load

# Sostituisci con il percorso del tuo file
modello = load("models/text_response_model.joblib")

print(modello.named_steps)
