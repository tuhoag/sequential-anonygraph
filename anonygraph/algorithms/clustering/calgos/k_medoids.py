import logging
import math
import numpy as np

from sklearn_extra.cluster import KMedoids
from .base_clustering_algo import BaseClusteringAlgo

import anonygraph.algorithms as algo

logger = logging.getLogger(__name__)

class KMedoidsClusteringAlgo(BaseClusteringAlgo):
    def __init__(self, min_size):
        self.__min_size = min_size
        # self.__max_size = max_size

    def run(self, entity_ids, pair_distance):
        num_clusters = int(len(entity_ids) / self.__min_size)

        if num_clusters > 1:
            # logger.debug("num clusters: {} ({}/{})".format(num_clusters, len(entity_ids), self.__min_size))
            distance_matrix = pair_distance.get_distance_matrix(entity_ids)

            algo_fn = KMedoids(n_clusters=num_clusters, init="k-medoids++", metric="precomputed")
            sk_clusters = algo_fn.fit_predict(distance_matrix)
            logger.debug(sk_clusters)

            clusters = convert_sklearn_clustering_results_to_cluster(sk_clusters, entity_ids)
        else:
            clusters = [algo.Cluster.from_iter(entity_ids)]

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