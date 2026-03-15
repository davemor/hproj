from dask.distributed import Client
from dask_cuda import LocalCUDACluster
    
def make_dask_client():
    cluster = LocalCUDACluster(
        threads_per_worker=1,
    )
    client = Client(cluster)
    return client, cluster