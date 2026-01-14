import torch
import torch.nn as nn
import csv
from pathlib import Path
from datetime import datetime

from sklearn.metrics import f1_score

from src.models import LSTMClassifier, HybridClassifier
from src.data.dataset import get_dataloaders
from src.training.train import CONFIG as TRAIN_CONFIG


def evaluate_with_f1(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if "Hybrid" in model.__class__.__name__:
                x_seq = X_batch[..., :-3]
                s_vec = X_batch[:, -1, -3:]
                outputs = model(x_seq, s_vec)
            else:
                outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    all_preds = torch.cat(all_preds).numpy().astype(int)
    all_targets = torch.cat(all_targets).numpy().astype(int)

    f1 = f1_score(all_targets, all_preds)
    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc, f1


def eval_on_test(model_type: str = "lstm"):
    cfg = TRAIN_CONFIG.copy()
    device = torch.device(cfg["device"])
    model_type = model_type.lower()

    print(f"\nEvaluating {model_type.upper()} model on TEST set")

    loaders = get_dataloaders(model_type=model_type, batch_size=cfg["batch_size"])
    test_loader = loaders["test"]

    # --- create model ---
    if model_type == "lstm":
        model = LSTMClassifier(
            input_size=3,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            bidirectional=cfg["bidirectional"],
        )
    else:
        model = HybridClassifier(
            input_size=3,
            sent_size=3,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            bidirectional=cfg["bidirectional"],
        )

    # --- load checkpoint ---
    ckpt_path = Path(__file__).resolve().parents[2] / "models" / "checkpoints" / f"{model_type}_best.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    # --- criterion ---
    criterion = nn.BCEWithLogitsLoss()

    # --- evaluate ---
    test_loss, test_acc, test_f1 = evaluate_with_f1(model, test_loader, criterion, device)

    print(f"[TEST] {model_type.upper()} → Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

    return model_type, test_loss, test_acc, test_f1


def save_results_to_csv(results):
    """Save evaluation results into a CSV file."""
    results_dir = Path(__file__).resolve().parents[2] / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "final_test_results.csv"
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        # Write header once
        if not file_exists:
            writer.writerow(["timestamp", "model_type", "test_loss", "test_acc", "test_f1"])

        for row in results:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp] + list(row))

    print(f"\nFinal results saved to: {csv_path}")


if __name__ == "__main__":
    results = []

    results.append(eval_on_test("lstm"))
    results.append(eval_on_test("hybrid"))

    save_results_to_csv(results)
