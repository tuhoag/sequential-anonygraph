import sys
import numpy as np
import logging
from sklearn_extra.cluster import KMedoids

import anonygraph.algorithms as algo

logger = logging.getLogger(__name__)

class SameSizeKMedoidsClustering(object):
    def __init__(self, num_clusters):
        self.__num_clusters = num_clusters

    def run(self, entity_ids, pair_distance):
        distace_matrix = pair_distance.get_distance_matrix(entity_ids)
        logger.debug(distace_matrix)

        algo_fn = KMedoids(n_clusters=self.__num_clusters, metric="precomputed", init="k-medoids++")
        algo_fn.fit(distace_matrix)

        medoids = list(map(lambda medoid_idx: entity_ids[medoid_idx], algo_fn.medoid_indices_))
        logger.debug("centers: {} -> {} in {}".format(algo_fn.medoid_indices_, medoids, entity_ids))
        # logger.debug(raw_clusters)

        user_cluster_dist = pair_distance.get_distance_matrix_of_entity_and_entity(entity_ids, medoids)
        logger.debug("user cluster dist {}: {}".format(user_cluster_dist.shape, user_cluster_dist))

        farthest_user_cluster_dist = np.max(user_cluster_dist, axis=0, keepdims=True)
        logger.debug("farthest cluster dist {}: {}".format(farthest_user_cluster_dist.shape, farthest_user_cluster_dist))

        delta_user_cluster_dist = user_cluster_dist - farthest_user_cluster_dist
        logger.debug('delta user cluster dist {}: {}'.format(delta_user_cluster_dist.shape, delta_user_cluster_dist))

        num_remaining_users = len(entity_ids)
        min_size = int(len(entity_ids) / self.__num_clusters)
        new_clusters = []

        for cluster_id in range(self.__num_clusters):
            logger.debug("cluster id: {}".format(cluster_id))

            orders = delta_user_cluster_dist[:, cluster_id].argsort()
            logger.debug("order: {} -> {}".format(delta_user_cluster_dist[:, cluster_id], orders))

            if num_remaining_users - min_size < min_size:
                num_selections = num_remaining_users
            else:
                num_selections = min_size

            chosen_idxes = orders[:num_selections]
            logger.debug('chosen idxes: {}'.format(chosen_idxes))

            chosen_dist = delta_user_cluster_dist[chosen_idxes, cluster_id]
            logger.debug('chosen dist: {}'.format(chosen_dist))


            delta_user_cluster_dist[chosen_idxes, :] = sys.maxsize
            logger.debug('after update: {}'.format(delta_user_cluster_dist))

            new_cluster = algo.Cluster()
            for chosen_user_idx in chosen_idxes:
                chosen_user_id = entity_ids[chosen_user_idx]
                logger.debug('user idx {} -> user id: {}'.format(chosen_user_idx, chosen_user_id))
                new_cluster.add_entity(chosen_user_id)

            logger.debug('new cluster: {}'.format(new_cluster))
            new_clusters.append(new_cluster)
            num_remaining_users = num_remaining_users - num_selections

        return new_clusters