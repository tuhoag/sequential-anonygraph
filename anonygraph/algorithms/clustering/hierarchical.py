import itertools
import logging
import sklearn.cluster as calgo
import sys

import anonygraph.algorithms as algo
import anonygraph.utils.general as utils

logger = logging.getLogger(__name__)


def convert_sklearn_clustering_results_to_cluster(clustering_results, pair_distance):
    results_dict = {}

    entity_ids = pair_distance.entity_ids

    for entity_idx, cluster_id in enumerate(clustering_results):
        cluster = results_dict.get(cluster_id)

        if cluster is None:
            cluster = algo.Cluster()
            results_dict[cluster_id] = cluster

        cluster.add_entity(entity_ids[entity_idx])

    return list(results_dict.values())

def convert_max_dist(max_dist, max_max_dist, min_max_dist):
    return (max_max_dist - min_max_dist) * max_dist + min_max_dist + sys.float_info.epsilon



class Finder(object):
    def __init__(self, min_size, max_size, max_dist, cluster_distance_fn):
        self.__distances = {}
        self.__min_size = min_size
        self.__max_size = max_size
        self.__max_dist = max_dist
        self.__cluster_distance_fn = cluster_distance_fn


    def run(self, cluster_ids, all_clusters):
        closest_pair = (sys.maxsize, None, None)

        for cluster1_idx in range(len(cluster_ids) - 1):
            for cluster2_idx in range(cluster1_idx + 1, len(cluster_ids)):
                cluster1_id = cluster_ids[cluster1_idx]
                cluster2_id = cluster_ids[cluster2_idx]

                if cluster1_id < cluster2_id:
                    key = (cluster1_id, cluster2_id)
                else:
                    key = (cluster2_id, cluster1_id)

                cluster1 = all_clusters[cluster1_id]
                cluster2 = all_clusters[cluster2_id]

                if len(cluster1) + len(cluster2) > self.__max_size:
                    continue

                distance = self.__distances.get(key)
                if distance is None:
                    logger.debug('clusters {}, {}'.format(cluster1, cluster2))
                    distance = self.__cluster_distance_fn(cluster1, cluster2)
                    self.__distances[key] = distance

                if distance > self.__max_dist:
                    continue

                if distance < closest_pair[0]:
                    closest_pair = (distance, cluster1_id, cluster2_id)

        return closest_pair

class ClustersDistance(object):
    def __init__(self, entity_distance):
        self.entity_distance = entity_distance

    def __call__(self, cluster1, cluster2):
        pairs = itertools.product(cluster1, cluster2)
        distance = max(map(lambda item: self.entity_distance.get_distance(item[0], item[1]), pairs))

        return distance

class HierarchicalClustering(object):
    def __init__(self, min_size, max_size, max_dist):
        self.__min_size = min_size
        self.__max_size = max_size
        self.__max_dist = max_dist

    def run(self, entity_ids, pair_distance):
        max_distance = pair_distance.max_distance
        min_distance = pair_distance.min_distance
        raw_max_dist = convert_max_dist(self.__max_dist, max_distance, min_distance)
        logger.debug('({}, {}): {} -> {}'.format(min_distance, max_distance, self.__max_dist, raw_max_dist))

        distance_matrix = pair_distance.get_distance_matrix(entity_ids)

        calgo_fn = calgo.AgglomerativeClustering(n_clusters=None, distance_max_dist=raw_max_dist, affinity="precomputed", linkage="complete")
        result = calgo_fn.fit_predict(distance_matrix)
        logger.debug(result)
        # is_finished = False
        # all_clusters = []
        # for entity_id in entity_ids:
        #     all_clusters.append([entity_id])

        # current_cluster_ids = list(range(len(all_clusters)))

        # clusters_distance_fn = ClustersDistance(pair_distance)
        # finder_fn = Finder(self.__min_size, self.__max_size, raw_max_dist, clusters_distance_fn)
        # while not is_finished:
        #     chosen_score, chosen_cluster1_id, chosen_cluster2_id = finder_fn.run(current_cluster_ids, all_clusters)
        #     logger.debug('choosen clusters: {}'.format((chosen_score, chosen_cluster1_id, chosen_cluster2_id)))

        #     # new_cluster = res




        #     raise Exception()
        return None