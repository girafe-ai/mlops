from typing import List
import time
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import faiss
from sklearn.model_selection import train_test_split


class FaissANN:
    def __init__(self, num_clusters: int, use_gpu: bool):
        self.num_clusters = num_clusters
        self.use_gpu = use_gpu

    def train(self, embeddings):
        embeddings = np.array(embeddings).astype(np.float32)
        faiss.normalize_L2(embeddings)

        d = embeddings.shape[1]
        if self.num_clusters > 0:
            quatnizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(
                quatnizer, d, self.num_clusters, faiss.METRIC_INNER_PRODUCT
            )
        else:
            self.index = faiss.IndexFlatIP(d)

        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 3, self.index
            )

        self.index.train(embeddings)
        self.index.add(embeddings)

    def search(self, query_embeddings, k: int):
        query_embeddings = np.array(query_embeddings, dtype=np.float32)
        faiss.normalize_L2(query_embeddings)
        D, I = self.index.search(query_embeddings, k)
        return D, I


def validate_accuracy(
    faiss_ann: FaissANN, train_df: pd.DataFrame, test_df: pd.DataFrame
):
    correct_predictions = 0
    total_predictions = len(test_df)

    begin = time.time()
    for emb, _, target in test_df.values:
        _, I = faiss_ann.search([emb], k=1)
        predicted_index = I[0][0]
        predicted_target = train_df.iloc[predicted_index]["target"]
        if predicted_target == target:
            correct_predictions += 1
    total_time = time.time() - begin
    throughput = int(total_predictions / total_time)

    return throughput, round(correct_predictions / total_predictions, 3)


def synthetic_throughput(faiss_ann: FaissANN, df: pd.DataFrame, msg: str):
    for emb, _, _ in tqdm(df.values, desc=msg):
        faiss_ann.search([emb], k=1)


def run_validation_tests(
    df_path, cluster_settings, test_size, use_gpu, emb_column: str = "emb"
):
    df = pd.read_parquet(df_path)
    df = df[[emb_column, "intent", "target"]]
    df.columns = ["emb", "intent", "target"]
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    for num_clusters in cluster_settings:
        faiss_ann = FaissANN(num_clusters, use_gpu=use_gpu)
        faiss_ann.train(train_df["emb"].tolist())
        throughput, acc = validate_accuracy(faiss_ann, train_df, test_df)
        print(
            f"Accuracy with clusters = {num_clusters}: {acc}. Throughput: {throughput}"
        )


def run_performance_tests(
    df_path: str,
    cluster_settings: List[int],
    size_limits: List[int],
    test_size,
    use_gpu: bool,
    emb_column: str,
):
    df = pd.read_parquet(df_path)
    df = df[[emb_column, "intent", "target"]]
    df.columns = ["emb", "intent", "target"]
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    for num_clusters, size_limit in zip(cluster_settings, size_limits):
        faiss_ann = FaissANN(num_clusters=num_clusters, use_gpu=use_gpu)
        faiss_ann.train(train_df["emb"].tolist())

        device = "GPU" if use_gpu else "CPU"
        synthetic_throughput(
            faiss_ann,
            test_df[:size_limit],
            msg=f"{device}, with clusters = {num_clusters}",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-column", type=str)
    emb_column = parser.parse_args().emb_column

    real_data_path = "./xlm-roberta-embeddings.parquet"
    run_validation_tests(
        real_data_path,
        cluster_settings=[0, 8, 16, 64, 128],
        test_size=0.2,
        emb_column=emb_column,
        use_gpu=True,
    )

    synthetic_data_path = "./xlm-roberta-synthetic-embeddings.parquet"
    run_performance_tests(
        synthetic_data_path,
        [0],
        [100],
        use_gpu=False,
        emb_column=emb_column,
        test_size=0.1,
    )

    clusters_perf = [0, 8, 16, 32, 64, 128, 512, 1024, 4096, 8192, 16384]
    test_size_limits = [
        1000,
        100,
        500,
        1000,
        1000,
        2000,
        20_000,
        20_000,
        20_000,
        20_000,
        100_000,
    ]
    run_performance_tests(
        synthetic_data_path,
        clusters_perf,
        test_size_limits,
        use_gpu=True,
        emb_column=emb_column,
        test_size=0.1,
    )


if __name__ == "__main__":
    main()
