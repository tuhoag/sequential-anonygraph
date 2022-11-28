import logging
import math
import numpy as np

from anonygraph.algorithms.cluster import Cluster


from .base_clustering_algo import BaseClusteringAlgo

import anonygraph.algorithms as algo

logger = logging.getLogger(__name__)

class CustomClusteringAlgo(BaseClusteringAlgo):
    def __init__(self, t, entity_name2id):
        self.t = t
        self.entity_name2id = entity_name2id

    def run(self, entity_ids, pair_distance):
        logger.debug("entity_ids: {}".format(entity_ids))
        logger.debug("entity_name2id: {}".format(self.entity_name2id))

        if self.t == 0:
            name_clusters = [
                ["user_0", "user_1"],
                ["user_2", "user_3", "user_4"],
                ["user_5"]
            ]
        elif self.t == 1:
            name_clusters = [
                ["user_6"]
            ]
        elif self.t == 2:
            name_clusters = [
                ["user_6", "user_7"]
            ]
        else:
            name_clusters = []

        clusters = []
        for name_cluster in name_clusters:
            cluster = Cluster.from_iter(map(lambda entity_name: self.entity_name2id[entity_name], name_cluster))

            clusters.append(cluster)

        return clusters