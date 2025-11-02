from pathlib import Path
import joblib
from sklearn.metrics import classification_report
from dataset import load_samples  # quello della tua repo
from loss_visualizaer import plot_loss_from_json

MODEL_PATH = Path("models/text_response_model.joblib")
DATASET_DIR = Path("zendod_dataset")
CONFIG = "massive"
SPLIT = "test"   # o "validation"

def main():
    # 1) carica il bundle (pipeline + label_encoder)
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    label_encoder = bundle["label_encoder"]

    # 2) carica i dati di test
    samples = load_samples(DATASET_DIR, config=CONFIG, split=SPLIT)
    texts = [s.prompt for s in samples]
    true_labels = [s.intent for s in samples]   # <-- queste sono STRINGHE

    # 3) predici (la pipeline ora restituisce ID numerici)
    y_pred_ids = pipeline.predict(texts)

    # 4) riconverti gli ID in etichette testuali
    pred_labels = label_encoder.inverse_transform(y_pred_ids)

    # 5) ora true_labels e pred_labels sono ENTRAMBE stringhe → ok
    print("\n📊 Risultati sullo split:", SPLIT)
    print(classification_report(true_labels, pred_labels, zero_division=0))
    plot_loss_from_json(json_path='models/training_log.json')


if __name__ == "__main__":
    main()