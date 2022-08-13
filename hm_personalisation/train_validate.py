from pathlib import Path
from typing import List, Optional, Tuple

import click
import numpy as np
import torch
from loguru import logger

from .checkpoints import TrainLogger
from .ml import ArrayBatchSampler, InteractionDataset, MatrixFactorization


def get_train_validation_indices(
    data: torch.utils.data.Dataset, validation_perc: float
) -> Tuple[List[int], List[int]]:
    indices = np.arange(data.records_frame.shape[0])
    np.random.shuffle(indices)
    validation_size = int(validation_perc * len(indices))
    return indices[:-validation_size], indices[-validation_size:]


def train(
    data: InteractionDataset,
    train_indices: List[int],
    validation_indices: List[int],
    num_latent_features: int,
    l2_lambda: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    model_name: str,
    model_version: str,
):
    model = MatrixFactorization(
        num_customers=data.num_customers,
        num_articles=data.num_articles,
        num_latent_features=num_latent_features,
        l2_lambda=l2_lambda,
    )
    customer_idxs_val, article_idxs_val, weights_val = (
        torch.from_numpy(arr) for arr in data[validation_indices]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_logger = TrainLogger(model_name, model_version)
    train_logger.save_model_params(
        {
            "train_filename": str(data.csv_path),
            "train_n_rows": data.n_rows,
            "num_customers": data.num_customers,
            "num_articles": data.num_articles,
            "num_latent_features": num_latent_features,
            "l2_lambda": l2_lambda,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
        }
    )
    train_logger.save_dataset_info(data)
    for j in range(epochs):
        train_loader = torch.utils.data.DataLoader(
            data, batch_size=None, sampler=ArrayBatchSampler(data, batch_size, train_indices)
        )
        train_pred_loss = 0

        for i, (customer_idxs, article_idxs, weights) in enumerate(train_loader):
            pred_loss = model.prediction_loss(customer_idxs, article_idxs, weights)
            penalty_loss = model.penalty_loss(customer_idxs, article_idxs)
            loss = pred_loss + penalty_loss
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Keep track of total loss across the whole training dataset
            train_pred_loss += pred_loss.detach().item()
            # Logging
            train_logger.log_batch(j, i, pred_loss.detach().item(), penalty_loss.detach().item())

        train_pred_loss /= len(train_indices)

        model.eval()
        with torch.no_grad():
            val_pred_loss = model.prediction_loss(
                customer_idxs_val, article_idxs_val, weights_val
            ).item()
            val_pred_loss /= len(validation_indices)
        train_logger.log_epoch(j, train_pred_loss, val_pred_loss)
        model.train()

        if (j + 1) % 10 == 0:
            logger.info(f"epoch {j} | train_pred_loss: {train_pred_loss:.4f}")
            logger.info(f"epoch {j} | val_pred_loss: {val_pred_loss:.4f}")
        if (j + 1) % 10 == 0 or j == epochs - 1:
            train_logger.save_model_checkpoint(model, optimizer, j, train_pred_loss, val_pred_loss)

    return model


@click.command()
@click.option("--model-name", type=str, required=True)
@click.option("--model-version", type=str, required=True)
@click.option("--train-filename", type=str, required=True)
@click.option("--n-rows", type=int)
@click.option("--val-perc", type=click.FloatRange(min=0.005, max=0.1), default=0.05)
@click.option("--num-latent-features", type=int, default=10)
@click.option("--l2-lambda", type=float, default=0.02)
@click.option("--learning-rate", type=float, default=0.001)
@click.option("--batch-size", type=int, default=100000)
@click.option("--epochs", type=int, default=200)
@click.option("--data-location", type=click.Choice(["local", "remote"]), default="local")
def main(
    model_name: str,
    model_version: str,
    train_filename: str,
    n_rows: Optional[int],
    val_perc: float,
    num_latent_features: int,
    l2_lambda: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    data_location: str,
):
    if data_location == "local":
        data_path = Path(__file__).resolve().parent.parent / "data"
    data = InteractionDataset(data_path / train_filename, n_rows=n_rows)
    train_indices, validation_indices = get_train_validation_indices(data, validation_perc=val_perc)
    train(
        data=data,
        train_indices=train_indices,
        validation_indices=validation_indices,
        num_latent_features=num_latent_features,
        l2_lambda=l2_lambda,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        model_name=model_name,
        model_version=model_version,
    )


if __name__ == "__main__":
    main()
