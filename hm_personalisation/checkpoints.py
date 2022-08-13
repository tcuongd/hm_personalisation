import copy
import csv
import json
from datetime import datetime
from pathlib import Path
import pickle

import torch

from .ml import InteractionDataset, MatrixFactorization

LOGS_PATH = Path(__file__).resolve().parent.parent / "training_logs"
SAVED_MODELS_PATH = Path(__file__).resolve().parent.parent / "saved_models"


class TrainLogger:
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.batch_logs_path = LOGS_PATH / f"{model_name}__{model_version}__batches.csv"
        self.epoch_logs_path = LOGS_PATH / f"{model_name}__{model_version}__epochs.csv"
        self.model_state_logs_path = LOGS_PATH / f"{model_name}__{model_version}__state.json"
        self.param_changes_logs_path = (
            LOGS_PATH / f"{model_name}__{model_version}__param_changes.csv"
        )
        self.model_checkpoint_dir = SAVED_MODELS_PATH / f"{model_name}__{model_version}"
        self.model_params_path = self.model_checkpoint_dir / "params.json"
        for f in [
            self.batch_logs_path,
            self.epoch_logs_path,
            self.model_state_logs_path,
            self.param_changes_logs_path,
            self.model_params_path,
        ]:
            if f.exists():
                f.unlink()
        self.model_checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def log_batch(
        self, epoch_number: int, batch_number: int, pred_loss: float, penalty_loss: float
    ) -> None:
        batch_info = {
            "timestamp": datetime.utcnow(),
            "epoch_number": epoch_number,
            "batch_number": batch_number,
            "pred_loss": pred_loss,
            "penalty_loss": penalty_loss,
            "loss": pred_loss + penalty_loss,
        }
        header_required = not self.batch_logs_path.exists()
        with open(self.batch_logs_path, "a") as f:
            writer = csv.DictWriter(f, fieldnames=batch_info.keys())
            if header_required:
                writer.writeheader()
            writer.writerow(batch_info)

    def log_epoch(self, epoch_number: int, train_pred_loss: float, val_pred_loss: float) -> None:
        epoch_info = {
            "timestamp": datetime.utcnow(),
            "epoch_number": epoch_number,
            "train_pred_loss": train_pred_loss,
            "val_pred_loss": val_pred_loss,
        }
        header_required = not self.epoch_logs_path.exists()
        with open(self.epoch_logs_path, "a") as f:
            writer = csv.DictWriter(f, fieldnames=epoch_info.keys())
            if header_required:
                writer.writeheader()
            writer.writerow(epoch_info)

    def log_model_state(
        self, model: torch.nn.Module, first_n_items: int = 10, decimal_places: float = 4
    ) -> None:
        state = {}
        for name, tensor in model.state_dict().items():
            state[name] = tensor.detach().numpy()[:first_n_items].round(decimal_places).tolist()
        with open(self.model_state_logs_path, "a") as f:
            json.dump(state, f)
            f.write("\n")

    def log_model_param_changes(
        self, epoch_number: int, batch_number: int, model: torch.nn.Module, tol: float = 0.0001
    ) -> None:
        if not hasattr(self, "current_params"):
            self.current_params = model.state_dict()
            return None
        else:
            changes = []
            new_params = model.state_dict()
            for name, tensor in new_params.items():
                row = {
                    "parameter": name,
                    "epoch_number": epoch_number,
                    "batch_number": batch_number,
                    "timestamp": datetime.utcnow(),
                }
                row["num_values_changed"] = (
                    ((tensor - self.current_params[name]).abs() > tol).sum().item()
                )
                changes.append(row)

            header_required = not self.param_changes_logs_path.exists()
            with open(self.param_changes_logs_path, "a") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=(
                        "timestamp",
                        "epoch_number",
                        "batch_number",
                        "parameter",
                        "num_values_changed",
                    ),
                )
                if header_required:
                    writer.writeheader()
                writer.writerows(changes)
            self.current_params = new_params

    def save_model_params(self, params: dict) -> None:
        with open(self.model_params_path, "w") as f:
            json.dump(params, f, indent=2)

    def save_model_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch_number: int,
        train_loss: float,
        val_loss: float,
    ) -> None:
        checkpoint = {
            "epoch": epoch_number,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, self.model_checkpoint_dir / f"epoch{epoch_number}.tar")

    def save_dataset_info(self, data: InteractionDataset) -> None:
        info = copy.deepcopy(data.__dict__)
        del info["records_frame"]
        with open(self.model_checkpoint_dir / "dataset_info.pkl", "wb") as f:
            pickle.dump(info, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(model_name: str, model_version: str, epoch_number: int):
    save_dir = Path(".") / "saved_models" / f"{model_name}__{model_version}"
    with open(save_dir / "params.json", "r") as f:
        params = json.load(f)
    checkpoint = torch.load(save_dir / f"epoch{epoch_number}.tar")
    with open(save_dir / "dataset_info.pkl", "rb") as f:
        dataset_info = pickle.load(f)

    model = MatrixFactorization(
        num_customers=params["num_customers"],
        num_articles=params["num_articles"],
        num_latent_features=params["num_latent_features"],
        l2_lambda=params["l2_lambda"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    checkpoint_info = {
        k: v for k, v in checkpoint.items() if k in ["epoch", "train_loss", "val_loss"]
    }
    return model, optimizer, checkpoint_info, dataset_info
