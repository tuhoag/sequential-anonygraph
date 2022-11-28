from anonygraph.algorithms.clustering.calgos.base_clustering_algo import BaseClusteringAlgo
from anonygraph.algorithms.clustering.calgos.hdbscan import HDBSCANClustering
from anonygraph.constants import *
from anonygraph.utils import data as dutils

from .k_medoids import KMedoidsClusteringAlgo
from .custom import CustomClusteringAlgo


def get_clustering_algorithm(algo_name, args):
    if algo_name == KMEDOIDS_CLUSTERING:
        min_size = args["k"]
        calgo_fn = KMedoidsClusteringAlgo(min_size)

    elif algo_name == HDBSCAN_CLUSTERING:
        min_size = args["k"]
        calgo_fn = HDBSCANClustering(min_size)

    elif algo_name == CUSTOM_CLUSTERING:
        data_name = args["data"]
        entity_name2id = dutils.get_raw_entity_indexes(
            data_name, args["sample"]
        )
        t = args["t"]
        if data_name != "dummy":
            raise Exception(
                "Custom Clustering Algo only supports dummy data set"
            )
        calgo_fn = CustomClusteringAlgo(t, entity_name2id)
    else:
        raise NotImplementedError(
            "Unsupported clustering algorithm: {}".format(algo_name)
        )

    return calgo_fn