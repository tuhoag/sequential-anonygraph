import logging
import numpy as np

import anonygraph.algorithms as algo

logger = logging.getLogger(__name__)

def choose_the_farthest_entity_id(entity_ids, pair_distance):
    pass

def choose_a_random_entity_id(entity_ids):
    return np.random.choice(entity_ids)

def assign_remaining_entities(clusters, remaining_entity_idxes, entity_ids, distance_matrix):

    pass

class NearestNeighborsClustering(object):
    def __init__(self, min_size, max_size, max_dist):
        self.__min_size = min_size
        self.__max_size = max_size
        self.__max_dist = max_dist

    def run(self, entity_ids, pair_distance):
        # choose the farthest entity
        current_entity_idxes = list(range(len(entity_ids)))

        distance_matrix = pair_distance.get_distance_matrix(entity_ids)

        logger.debug(distance_matrix)

        clusters = algo.Clusters()

        logger.debug("entity ids: {}".format(entity_ids))

        while len(current_entity_idxes) > self.__min_size:
            current_index = np.random.randint(len(current_entity_idxes))
            current_entity_idx = current_entity_idxes[current_index]
            logger.debug("first entity id: {}".format(current_entity_idx))

            current_dist_matrix = distance_matrix[current_entity_idx, :]
            logger.debug(current_dist_matrix)

            orders = current_dist_matrix.argsort()
            logger.debug(orders)

            num_selection = self.__min_size

            chosen_idxes = orders[:num_selection]
            logger.debug(chosen_idxes)


            new_cluster = algo.Cluster()
            for chosen_entity_idx in chosen_idxes:
                chosen_entity_id = entity_ids[chosen_entity_idx]
                new_cluster.add_entity(chosen_entity_id)

                logger.debug('chosen idx: {} -> id: {} in {}'.format(chosen_entity_idx, chosen_entity_id, current_entity_idxes))
                current_entity_idxes.remove(chosen_entity_idx)
                # del current_entity_idxes[chosen_entity_idx]

            logger.debug("new cluster: {}".format(new_cluster))
            clusters.add_cluster(new_cluster)

        raise Exception()

        logger.debug("remaining entity idxes: {}".format(current_entity_idxes))

        logger.debug(clusters)

        return [entity_ids]