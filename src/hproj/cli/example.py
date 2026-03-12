import click
from cuml.dask.cluster import KMeans
from cuml.dask.datasets import make_blobs
from cuml.metrics import adjusted_rand_score
from dask.distributed import Client
from dask_cuda import LocalCUDACluster


@click.command()
def example():
    print("Running the pipeline")

    # set up a local CUDA cluster and connect a Dask client to it
    cluster = LocalCUDACluster()
    client = Client(cluster)

    # for remote manchines, remember to do ssh port forwarding
    print(f"Cluster dashboard available at: {client.dashboard_link}")

    # Get number of workers for data partitioning
    n_workers = len(client.scheduler_info()["workers"])

    # Generate distributed synthetic data
    X, y = make_blobs(
        n_samples=10000,
        n_features=20,
        centers=5,
        cluster_std=0.5,
        random_state=42,
        n_parts=n_workers * 2,  # Multiple partitions per worker
    )

    print(f"Generated data with {len(X.to_delayed())} partitions")
    print(f"Data type: {type(X)}")

    # Train distributed K-Means
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)

    # Make predictions
    labels = kmeans.predict(X)

    # Evaluate clustering quality
    score = adjusted_rand_score(y.compute(), labels.compute())
    print(f"Adjusted Rand Score: {score:.4f}")

    # View cluster centers
    print(f"\nCluster centers shape: {kmeans.cluster_centers_.shape}")
