import click
import polars as pl
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import List, Dict
from loguru import logger

from .checkpoints import load_checkpoint
from .evaluation import apk


def get_existing_customers(
    modelled_customer_ints: List[int], holdout_customer_ints: List[int]
) -> List[int]:
    """Find the customers in the holdout set which were also modelled. Duplicates in either list are OK."""
    return np.intersect1d(
        np.unique(np.array(modelled_customer_ints)),
        np.unique(np.array(holdout_customer_ints)),
    )


class Recommender:
    def __init__(self, model_name: str, model_version: str, epoch_number: int):
        self.model_name = model_name
        self.model_version = model_version
        self.epoch_number = epoch_number
        self.model, _, _, self.dataset_info = load_checkpoint(
            model_name, model_version, epoch_number
        )
        self.model.eval()

    def predict_weights_existing(self, customer_int: int) -> np.array:
        """For a customer_int present in the model, predict their weights for every article."""
        articles = torch.arange(self.model.num_articles)
        customer_idx = self.dataset_info["customer_int_to_idx"][customer_int]
        with torch.no_grad():
            preds = self.model.predicted_matrix(torch.tensor(customer_idx), articles)
        return preds.numpy()[0, :]

    def predict_weights_new(self, sample_size: int = 1000) -> np.array:
        """For the 'average' customer, predict their weights for every article."""
        sample_customer_idxs = np.random.choice(
            np.arange(self.model.num_customers), size=sample_size, replace=False
        )
        articles = torch.arange(self.model.num_articles)
        preds = []
        with torch.no_grad():
            for customer_idx in sample_customer_idxs:
                sample_weights = self.model.predicted_matrix(
                    torch.tensor(customer_idx), articles
                ).numpy()[0, :]
                preds.append(sample_weights)
        return np.vstack(preds).mean(axis=0)

    def apply_rules(
        self,
        customer_int: int,
        predicted_weights: np.array,
        purchase_history: pl.DataFrame,
        purchase_cutoffs: pl.DataFrame,
        purchase_cutoff_percentile: str,
    ) -> List[float]:
        customer_purchase_history = purchase_history.filter(pl.col("customer_int") == customer_int)
        if customer_purchase_history.is_empty():
            return predicted_weights
        else:
            excl_article_ints = (
                customer_purchase_history.join(purchase_cutoffs, on="article_int", how="inner")
                .filter(pl.col("days_since_last_purchase") <= pl.col(purchase_cutoff_percentile))
                .get_column("article_int")
                .to_numpy()
            )
            excl_article_idxs = [
                self.dataset_info["article_int_to_idx"].get(a_int) for a_int in excl_article_ints
            ]
            excl_article_idxs = [idx for idx in excl_article_idxs if idx is not None]
            adjusted_weights = predicted_weights.copy()
            adjusted_weights[excl_article_idxs] = -1.0
            return adjusted_weights

    def get_top_article_ints(self, adjusted_weights: List[float], top_n: int) -> List[int]:
        top_idxs = np.argsort(np.array(adjusted_weights))[-top_n:][::-1]
        return [int(self.dataset_info["article_idx_to_int"][idx]) for idx in top_idxs]

    def get_recommendations(
        self,
        customer_int: int,
        existing_customer_ints: List[int],
        purchase_history: pl.DataFrame,
        purchase_cutoffs: pl.DataFrame,
        purchase_cutoff_percentile: str,
        default_weights: List[float],
        top_n: int,
    ) -> List[int]:
        if customer_int in existing_customer_ints:
            predicted_weights = self.predict_weights_existing(customer_int)
            predicted_weights = self.apply_rules(
                customer_int,
                predicted_weights,
                purchase_history,
                purchase_cutoffs,
                purchase_cutoff_percentile,
            )
        else:
            predicted_weights = default_weights
        return self.get_top_article_ints(predicted_weights, top_n=top_n)


def calculate_apk(holdout: pl.DataFrame, k: int) -> pl.DataFrame:
    """`holdout` must contain columns 'actuals' and 'recommendations', both lists of article_ints"""
    apks = [
        apk(actual=actuals, predicted=recommendations, k=k)
        for actuals, recommendations in zip(
            holdout.get_column("actuals"), holdout.get_column("recommendations")
        )
    ]
    return holdout.with_column(pl.Series(apks, dtype=pl.Float64).alias("apk"))


def format_predictions(
    holdout: pl.DataFrame, customer_map: Dict[str, int], article_map: Dict[str, int]
) -> pl.DataFrame:
    """Format from a list of actual and recommended article_ints to a space-separated string containing article_ids"""
    reverse_customer_map = {c_int: c_id for c_id, c_int in customer_map.items()}
    reverse_article_map = {a_int: a_id for a_id, a_int in article_map.items()}
    customer_ids = [reverse_customer_map[c_int] for c_int in holdout.get_column("customer_int")]
    predictions = [
        " ".join([reverse_article_map[a_int] for a_int in ordered_recs])
        for ordered_recs in holdout.get_column("recommendations")
    ]
    return pl.DataFrame({"customer_id": customer_ids, "prediction": predictions})


@click.command()
@click.option("--model-name", type=str, required=True)
@click.option("--model-version", type=str, required=True)
@click.option("--epoch-number", type=click.IntRange(min=0), required=True)
@click.option("--holdout-filename", type=str, default="dev/holdout.csv")
@click.option(
    "--purchase-history-filename", type=str, default="dev/customer_days_since_last_purchase.csv"
)
@click.option("--purchase-cutoffs-filename", type=str, default="dev/days_between_purchase.csv")
@click.option(
    "--purchase-cutoff-percentile",
    type=click.Choice(["p10", "p25", "p50", "p75", "p90"]),
    default="p50",
)
@click.option("--evaluation-k", type=click.IntRange(min=1), default=12)
@click.option("--data-location", type=click.Choice(["local", "remote"]), default="local")
def main(
    model_name: str,
    model_version: str,
    epoch_number: int,
    holdout_filename: str,
    purchase_history_filename: str,
    purchase_cutoffs_filename: str,
    purchase_cutoff_percentile: str,
    evaluation_k: int,
    data_location: str,
):
    if data_location == "local":
        data_dir = Path(__file__).resolve().parent.parent / "data"
    holdout_raw = pl.read_csv(data_dir / holdout_filename)
    purchase_history = pl.read_csv(data_dir / purchase_history_filename)
    purchase_cutoffs = pl.read_csv(data_dir / purchase_cutoffs_filename)

    holdout = holdout_raw.groupby("customer_int").agg(pl.col("article_int").list().alias("actuals"))
    recommender = Recommender(
        model_name=model_name, model_version=model_version, epoch_number=epoch_number
    )
    existing_customers = get_existing_customers(
        list(recommender.dataset_info["customer_int_to_idx"].keys()),
        holdout.get_column("customer_int").to_numpy(),
    )
    logger.info("Getting fallback prediction")
    fallback = recommender.predict_weights_new()

    all_recs = []
    logger.info(
        f"Getting recommendations: {len(existing_customers):,.0f} existing users, {holdout.get_column('customer_int').shape[0]:,.0f} total"
    )
    for customer_int in holdout.get_column("customer_int"):
        rec = recommender.get_recommendations(
            customer_int=customer_int,
            existing_customer_ints=existing_customers,
            purchase_history=purchase_history,
            purchase_cutoffs=purchase_cutoffs,
            purchase_cutoff_percentile=purchase_cutoff_percentile,
            default_weights=fallback,
            top_n=evaluation_k,
        )
        all_recs.append(rec)
    holdout = holdout.with_column(pl.Series(all_recs).alias("recommendations"))

    predictions_dir = (
        Path(__file__).resolve().parent.parent
        / "predictions"
        / ("dev" if "dev" in holdout_filename else "submission")
        / f"{model_name}__{model_version}"
    )
    predictions_dir.mkdir(exist_ok=True, parents=True)

    if "dev" in holdout_filename:
        holdout = calculate_apk(holdout, evaluation_k)
        logger.info(f"Holdout MAP@{evaluation_k}: {holdout.get_column('apk').mean():.4f}")
        holdout.select(["customer_int", "apk"]).write_csv(predictions_dir / "holdout_apks.csv")
        logger.info(f"User-level APKs saved to {predictions_dir}")

    with open(data_dir / "customer_map.pkl", "rb") as f:
        customer_map = pickle.load(f)
    with open(data_dir / "article_map.pkl", "rb") as f:
        article_map = pickle.load(f)
    holdout_formatted = format_predictions(holdout, customer_map, article_map)
    holdout_formatted.write_csv(predictions_dir / "submission.csv")
    logger.info(f"Submission file saved to {predictions_dir}")


if __name__ == "__main__":
    main()
