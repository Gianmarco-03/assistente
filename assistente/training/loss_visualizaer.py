import json
from pathlib import Path
import matplotlib.pyplot as plt


def plot_loss_from_list(losses: list[dict], output_file: str | Path | None = None, show: bool = True):
    """
    losses: lista di dict tipo:
        [{"epoch": 1, "train_loss": 0.9},
         {"epoch": 2, "train_loss": 0.6}, ...]
    """
    if not losses:
        print("⚠️ Nessuna loss da plottare.")
        return

    epochs = [e["epoch"] for e in losses]
    train_loss = [e["train_loss"] for e in losses]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title("Andamento della Loss durante l'allenamento")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_file is not None:
        output_file = Path(output_file)
        plt.savefig(output_file)
        print(f"✅ Plot salvato in: {output_file}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_loss_from_json(json_path: str | Path, output_file: str | Path | None = None, show: bool = True):
    """
    Versione che legge dal json salvato a fine training.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"File JSON {json_path} non trovato.")

    with json_path.open("r", encoding="utf-8") as f:
        losses = json.load(f)

    plot_loss_from_list(losses, output_file=output_file, show=show)


if __name__ == "__main__":
    # uso da linea di comando:
    # python -m assistente.training.plot_loss
    default_json = Path("models") / "training_log.json"
    if default_json.exists():
        plot_loss_from_json(default_json, output_file=Path("models") / "loss_plot.png", show=True)
    else:
        print("⚠️ Nessun training_log.json trovato in ./models")