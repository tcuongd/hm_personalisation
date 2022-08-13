from pathlib import Path
from typing import Iterator, List, Optional, Sized, Tuple, Union

import numpy as np
import polars as pl
import torch


class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: Path, n_rows: Optional[int] = None):
        """
        Note that this custom Dataset is only compatible with:
            - DataLoader with automatic batching disabled (batch_size=None)
            - The custom sampler ArrayBatchSampler
        """
        self.csv_path = csv_path
        self.n_rows = n_rows
        self.records_frame = pl.read_csv(csv_path, n_rows=n_rows)
        self.customer_int_to_idx = {
            c_int: idx
            for c_int, idx in zip(
                np.sort(self.records_frame["customer_int"].unique()),
                range(len(self.records_frame["customer_int"].unique())),
            )
        }
        self.customer_idx_to_int = {v: k for k, v in self.customer_int_to_idx.items()}
        self.article_int_to_idx = {
            a_int: idx
            for a_int, idx in zip(
                np.sort(self.records_frame["article_int"].unique()),
                range(len(self.records_frame["article_int"].unique())),
            )
        }
        self.article_idx_to_int = {v: k for k, v in self.article_int_to_idx.items()}
        self.num_customers = len(self.customer_int_to_idx)
        self.num_articles = len(self.article_int_to_idx)

    def __len__(self) -> int:
        return self.records_frame.shape[0]

    def __getitem__(
        self, idx: Union[torch.tensor, np.array, int]
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Given a row indices, get (customer_idx, article_idx) pairs with their weight value, then return
        the results as separate arrays, i.e. one array for customer_idx, one array for article_idx,
        one array for weight values.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        records_frame_batch = self.records_frame[idx, :]
        return (
            np.array(
                [
                    self.customer_int_to_idx[c_int]
                    for c_int in records_frame_batch.get_column("customer_int")
                ]
            ),
            np.array(
                [
                    self.article_int_to_idx[a_int]
                    for a_int in records_frame_batch.get_column("article_int")
                ]
            ),
            records_frame_batch.get_column("weight").to_numpy().copy(),
        )

    def lookup_weight_individual(self, customer_idx: int, article_idx: int) -> float:
        """Pass in the (sorted) index values for one customer and one article. Return the actual weight,
        or np.nan if the pair does not exist in the historical data. Note that you should not pass the
        actual customer or article IDs.
        """
        return self.records_frame.filter(
            (pl.col("customer_int") == self.customer_idx_to_int[customer_idx])
            & (pl.col("article_int") == self.article_idx_to_int[article_idx])
        ).get_column("weight")[0]

    def lookup_weight_batch(self, customer_idx: List[int], article_idx: List[int]) -> List[float]:
        return [
            self.lookup_weight_individual(c_idx, a_idx)
            for c_idx, a_idx in zip(customer_idx, article_idx)
        ]


class ArrayBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self, data_source: Sized, batch_size: int, subset_indices: Optional[List[int]] = None
    ):
        """
        Parameters
        ----------
        data_source: A Dataset to sample from
        batch_size: The number of samples per batch
        subset_indices: Only consider this subset of indices for sampling. Used for training / validation
            splits. Default None, which considers all indices in the data source.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.subset_indices = subset_indices

    def get_batches(self) -> Iterator[List[int]]:
        if self.subset_indices is None:
            subset_indices = np.arange(len(self.data_source), dtype=int)
        else:
            subset_indices = np.array(self.subset_indices)
        np.random.shuffle(subset_indices)
        return torch.from_numpy(subset_indices).split(self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.get_batches())

    def __len__(self) -> int:
        return len(self.get_batches())


class MatrixFactorization(torch.nn.Module):
    def __init__(
        self,
        num_customers: int,
        num_articles: int,
        num_latent_features: int,
        l2_lambda: float,
    ):
        super().__init__()
        self.num_customers = num_customers
        self.num_articles = num_articles
        self.num_latent_features = num_latent_features
        self.customer_embeddings = torch.nn.Embedding(
            num_embeddings=num_customers, embedding_dim=num_latent_features
        )
        self.customer_intercepts = torch.nn.Embedding(num_embeddings=num_customers, embedding_dim=1)
        self.article_embeddings = torch.nn.Embedding(
            num_embeddings=num_articles, embedding_dim=num_latent_features
        )
        self.article_intercepts = torch.nn.Embedding(num_embeddings=num_articles, embedding_dim=1)
        self.global_intercept = torch.nn.Parameter(torch.randn(1))
        self.l2_lambda = l2_lambda

    @staticmethod
    def forward(
        customer_embeddings: torch.Tensor,
        article_embeddings: torch.Tensor,
        customer_intercepts: torch.Tensor,
        article_intercepts: torch.Tensor,
        global_intercept: torch.Tensor,
    ) -> torch.Tensor:
        return torch.exp(
            torch.matmul(customer_embeddings, article_embeddings.T)
            + torch.reshape(customer_intercepts, (len(customer_intercepts), 1))
            + torch.reshape(article_intercepts, (1, len(article_intercepts)))
            + global_intercept
        )

    def predicted_matrix(
        self, customer_idx: torch.Tensor, article_idx: torch.Tensor
    ) -> torch.Tensor:
        customers = customer_idx.unique(sorted=True)
        articles = article_idx.unique(sorted=True)
        customer_embeddings = self.customer_embeddings(customers)
        article_embeddings = self.article_embeddings(articles)
        customer_intercepts = self.customer_intercepts(customers)
        article_intercepts = self.article_intercepts(articles)
        return self.forward(
            customer_embeddings,
            article_embeddings,
            customer_intercepts,
            article_intercepts,
            self.global_intercept,
        )

    @staticmethod
    def actual_matrix(
        customer_idx: torch.Tensor, article_idx: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        customers = customer_idx.unique(sorted=True).tolist()
        customer_coord_map = {c_int: idx for c_int, idx in zip(customers, range(len(customers)))}
        articles = article_idx.unique(sorted=True).tolist()
        article_coord_map = {a_int: idx for a_int, idx in zip(articles, range(len(articles)))}
        interaction_matrix = torch.sparse_coo_tensor(
            indices=[
                [customer_coord_map[c] for c in customer_idx.tolist()],
                [article_coord_map[a] for a in article_idx.tolist()],
            ],
            values=weights,
        ).to_dense()
        return interaction_matrix

    def prediction_loss(
        self,
        customer_idx: torch.Tensor,
        article_idx: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the total squared prediction loss for the given customer_idx and article_idx pairs."""
        prediction = self.predicted_matrix(customer_idx, article_idx)
        actual = self.actual_matrix(customer_idx, article_idx, targets)
        non_zeros = actual > 0
        return torch.sum(torch.pow(prediction[non_zeros] - actual[non_zeros], 2))

    def penalty_loss(self, customer_idx: torch.Tensor, article_idx: torch.Tensor) -> torch.Tensor:
        """Returns the regularisation penalty for the given customer_idx and article_idx pairs."""
        customers = customer_idx.unique(sorted=True)
        articles = article_idx.unique(sorted=True)
        return self.l2_lambda * (
            torch.linalg.norm(self.customer_embeddings(customers))
            + torch.linalg.norm(self.article_embeddings(articles))
        )
