import pickle
from datetime import timedelta
from pathlib import Path
from typing import Dict, Tuple

import click
import numpy as np
import polars as pl
from loguru import logger

DATA_PATH = (Path(__file__).parent / ".." / "data").resolve()


def read_customers() -> pl.LazyFrame:
    return pl.scan_csv(DATA_PATH / "customers.csv").with_column(
        pl.col("customer_id").cumcount().alias("customer_int")
    )


def read_articles() -> pl.LazyFrame:
    return pl.scan_csv(DATA_PATH / "articles.csv", dtypes={"article_id": pl.Utf8}).with_column(
        pl.col("article_id").cumcount().alias("article_int")
    )


def generate_customer_map(customers: pl.LazyFrame) -> Dict[str, int]:
    customers = customers.select(pl.col(["customer_id", "customer_int"])).collect()
    return {c_id: c_int for c_id, c_int in zip(customers["customer_id"], customers["customer_int"])}


def generate_article_map(articles: pl.LazyFrame) -> Dict[str, int]:
    articles = articles.select(pl.col(["article_id", "article_int"])).collect()
    return {a_id: a_int for a_id, a_int in zip(articles["article_id"], articles["article_int"])}


def read_transactions(customers: pl.LazyFrame, articles: pl.LazyFrame) -> pl.LazyFrame:
    """
    Parameters
    ----------
    customers: A mapping from customer_id to customer_int
    articles: A mapping from article_id to article_int

    Mapping to integers is done to save space and enable faster lookups.

    Returns
    -------
    The transactions data with a cleaned date field and customer_id / article_id replaced with their
    integer values.
    """
    customers = customers.select(pl.col(["customer_id", "customer_int"]))
    articles = articles.select(pl.col(["article_id", "article_int"]))
    return (
        pl.scan_csv(DATA_PATH / "transactions_train.csv", dtypes={"article_id": pl.Utf8})
        .with_column(pl.col("t_dat").str.strptime(pl.Date, "%Y-%m-%d"))
        .join(customers, on="customer_id")
        .join(articles, on="article_id")
        .drop(columns=["customer_id", "article_id"])
    )


def split_transactions(
    transactions: pl.LazyFrame, holdout_days: int
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    max_date = transactions.select(pl.col("t_dat").max()).collect()[0, 0]
    train = transactions.filter(pl.col("t_dat") <= max_date - timedelta(days=holdout_days))
    holdout = transactions.filter(pl.col("t_dat") > max_date - timedelta(days=holdout_days))
    return train, holdout


def get_max_prices(train: pl.LazyFrame) -> pl.LazyFrame:
    """
    Using the training period data, obtain the canonical price of each article that was purchased. We
    proxy this with the max price.

    Returns
    -------
    A dict mapping the article_int (if it was purchased during the train period) to the max price.
    """
    df = train.groupby("article_int").agg(pl.col("price").max().alias("max_price")).collect()
    return {aint: price for aint, price in zip(df["article_int"], df["max_price"])}


def summarise_train(
    train: pl.LazyFrame,
    t_discount_strength: float,
    prices: Dict[int, float],
    price_discount_strength: float,
) -> pl.LazyFrame:
    """
    Transformations to the transaction data to arrive at the "value" of each user-item interaction:
        - transactions long ago are weighted down. t_discount_factor determines the strength of this
            downweighting. Set t_discount_strength = 0.0 for no downweighting, higher values to increase
            downweighting.
        - purchase prices are compared to their canonical price. Purchases for less than the canonical
            price are weighted down by multiplying by [price_discount_strength * price / max_price] + (1 - price_discount_strength).
            Set price_discount_strength = 0.0 for no downweighting, 1.0 for proportional downweighting.
        - multiple interactions for the same user-item interaction are added together, after weighting
            is performed.

    Returns
    -------
    Dataframe unique at the customer_int + article_int level, with the column `weight` representing
    the value of the interaction (higher = stronger and more relevant interaction). `weight` is always
    positive.
    """
    return (
        train.with_column(
            pl.col("article_int").apply(lambda x: prices.get(x, np.nan)).alias("max_price")
        )
        .with_column(
            (pl.col("price") / pl.col("max_price")).fill_nan(1.0).alias("price_proportion")
        )
        .with_column(
            (
                pl.col("price_proportion") * price_discount_strength + (1 - price_discount_strength)
            ).alias("price_weight")
        )
        .with_column(
            (pl.col("t_dat").max().dt.epoch_days() - pl.col("t_dat").dt.epoch_days()).alias(
                "days_from_today"
            )
        )
        .with_column(np.exp(-t_discount_strength * pl.col("days_from_today")).alias("time_weight"))
        .with_column((pl.col("price_weight") * pl.col("time_weight")).alias("weight"))
        .groupby(["customer_int", "article_int"])
        .agg([pl.col("weight").sum()])
        .filter(pl.col("weight") > 0)
    )


def summarise_holdout(holdout: pl.LazyFrame) -> pl.LazyFrame:
    """A dataframe in long form for the customer_ints who purchased at least one item in the holdout period."""
    return holdout.groupby(["customer_int", "article_int"]).agg([pl.count()])


def calculate_days_between_purchase(train: pl.LazyFrame) -> pl.LazyFrame:
    """Calculate the distribution of the number of days between repeat purchases for each article_int."""
    return (
        train.sort(["customer_int", "article_int"])
        .with_column(
            pl.col("t_dat").shift().over(["customer_int", "article_int"]).alias("t_dat_previous")
        )
        .filter(pl.col("t_dat_previous").is_not_null())
        .with_column(
            (pl.col("t_dat").dt.epoch_days() - pl.col("t_dat_previous").dt.epoch_days()).alias(
                "days_since_last_purchase"
            )
        )
        .filter(pl.col("days_since_last_purchase") > 0)
        .groupby("article_int")
        .agg(
            [
                pl.col("days_since_last_purchase").quantile(quantile=0.10).alias("p10"),
                pl.col("days_since_last_purchase").quantile(quantile=0.25).alias("p25"),
                pl.col("days_since_last_purchase").quantile(quantile=0.5).alias("p50"),
                pl.col("days_since_last_purchase").quantile(quantile=0.75).alias("p75"),
                pl.col("days_since_last_purchase").quantile(quantile=0.90).alias("p90"),
            ]
        )
    )


def get_customer_days_since_last_purchase(train: pl.LazyFrame) -> pl.LazyFrame:
    return (
        train.groupby(["customer_int", "article_int"])
        .agg(pl.col("t_dat").max().alias("last_purchase_date"))
        .with_column(
            (
                pl.col("last_purchase_date").max().dt.epoch_days() - pl.col("last_purchase_date")
            ).alias("days_since_last_purchase")
        )
        .select(["customer_int", "article_int", "days_since_last_purchase"])
    )


def get_transactions_first_n_days(train: pl.LazyFrame, first_n_days: int) -> pl.LazyFrame:
    """
    Get the items bought by users in their first N days, to train a separate statistical model to
    predict item preference for new users who weren't part of the collaborative filtering training.
    """
    return (
        train.with_column(pl.col("t_dat").first().over("customer_int").alias("t_dat_first"))
        .with_column(
            (pl.col("t_dat").dt.epoch_days() - pl.col("t_dat_first").dt.epoch_days()).alias(
                "days_since_first_t"
            )
        )
        .filter(pl.col("days_since_first_t") < first_n_days)
    )


def get_holdout_new_user_transactions(train: pl.LazyFrame, holdout: pl.LazyFrame) -> pl.LazyFrame:
    """Get the transactions in the holdout set relating to user who weren't seen in the training set."""
    train_users = train.select(pl.col("customer_int").unique()).collect()
    return holdout.filter(pl.col("customer_int").is_in(train_users["customer_int"]).is_not())


def prepare_lookups() -> None:
    customers = read_customers()
    articles = read_articles()

    customer_map = generate_customer_map(customers)
    with open(DATA_PATH / "customer_map.pkl", "wb") as f:
        pickle.dump(customer_map, f, pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved customer map to {DATA_PATH / 'customer_map.pkl'}")

    article_map = generate_article_map(articles)
    with open(DATA_PATH / "article_map.pkl", "wb") as f:
        pickle.dump(article_map, f, pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved article map to {DATA_PATH / 'article_map.pkl'}")


def prepare_training_data(
    holdout_days: int,
    t_discount_strength: float,
    price_discount_strength: float,
    new_user_days: int,
) -> None:
    savedir = "dev" if holdout_days > 0 else "submission"
    (DATA_PATH / savedir).mkdir(exist_ok=True, parents=True)

    customers = read_customers()
    articles = read_articles()

    transactions = read_transactions(customers, articles)
    train, holdout = split_transactions(transactions, holdout_days=holdout_days)

    prices = get_max_prices(train)
    with open(DATA_PATH / savedir / "prices.pkl", "wb") as f:
        pickle.dump(prices, f, pickle.HIGHEST_PROTOCOL)

    days_between_purchase = calculate_days_between_purchase(train)
    days_between_purchase.collect().to_csv(DATA_PATH / savedir / "days_between_purchase.csv")
    logger.info(f"Saved days between purchase data to {DATA_PATH / savedir}")

    customer_days_last_purchase = get_customer_days_since_last_purchase(train)
    customer_days_last_purchase.collect().to_csv(
        DATA_PATH / savedir / "customer_days_since_last_purchase.csv"
    )
    logger.info(f"Saved customer-article last purchase data to {DATA_PATH / savedir}")

    train_summary = summarise_train(
        train,
        t_discount_strength=t_discount_strength,
        prices=prices,
        price_discount_strength=price_discount_strength,
    )
    t_discount_strength_fmt = str(t_discount_strength).replace(".", "-")
    price_discount_strength_fmt = str(price_discount_strength).replace(".", "-")
    train_summary_fname = f"train__t_discount={t_discount_strength_fmt}__price_discount={price_discount_strength_fmt}.csv"
    train_summary.collect().to_csv(DATA_PATH / savedir / train_summary_fname)
    logger.info(f"Saved {train_summary_fname} to {DATA_PATH / savedir}")

    new_transactions_train = get_transactions_first_n_days(train, first_n_days=new_user_days)
    train_new_users_fname = f"train_new_users__first_n_days={int(new_user_days)}.csv"
    new_transactions_train.collect().to_csv(DATA_PATH / savedir / train_new_users_fname)
    logger.info(f"Saved {train_new_users_fname} to {DATA_PATH / savedir}")

    if holdout_days > 0:
        holdout_summary = summarise_holdout(holdout)
        new_transactions_holdout = get_holdout_new_user_transactions(train, holdout)
        holdout_summary.collect().to_csv(DATA_PATH / savedir / "holdout.csv")
        new_transactions_holdout.collect().to_csv(DATA_PATH / savedir / "holdout_new_users.csv")
        logger.info(f"Saved holdout data to {DATA_PATH / savedir}")


@click.command()
@click.argument("dataset", type=click.Choice(["lookups", "training", "submission"]))
@click.option("--holdout-days", type=click.IntRange(min=0), default=7)
@click.option("--t-discount-strength", type=click.FloatRange(min=0.0), default=0.0)
@click.option("--price-discount-strength", type=click.FloatRange(min=0.0, max=1.0), default=0.0)
@click.option("--new-user-days", type=click.IntRange(min=0), default=7)
def main(
    dataset: str,
    holdout_days: int,
    t_discount_strength: float,
    price_discount_strength: float,
    new_user_days: int,
):
    if dataset == "lookups":
        prepare_lookups()
    elif dataset == "training":
        prepare_training_data(
            holdout_days, t_discount_strength, price_discount_strength, new_user_days
        )
    elif dataset == "submission":
        prepare_training_data(0, t_discount_strength, price_discount_strength, new_user_days)


if __name__ == "__main__":
    main()
