import logging
import math
import numpy as np
import hdbscan

from .base_clustering_algo import BaseClusteringAlgo
import anonygraph.algorithms as algo

logger = logging.getLogger(__name__)

class HDBSCANClustering(BaseClusteringAlgo):
    def __init__(self, min_size):
        self.__min_size = min_size

    def run(self, entity_ids, pair_distance):
        distance_matrix = pair_distance.get_distance_matrix(entity_ids)

        algo_fn = hdbscan.HDBSCAN(min_cluster_size=self.__min_size, metric="precomputed")
        algo_fn.fit(distance_matrix)

        sk_clusters = algo_fn.labels_

        clusters = convert_sklearn_clustering_results_to_cluster(sk_clusters, entity_ids)

        return clusters


def convert_sklearn_clustering_results_to_cluster(clustering_results, entity_ids):
    results_dict = {}

    for entity_idx, cluster_id in enumerate(clustering_results):
        cluster = results_dict.get(cluster_id)

        if cluster is None:
            cluster = algo.Cluster()
            results_dict[cluster_id] = cluster

        cluster.add_entity(entity_ids[entity_idx])

    return list(results_dict.values())