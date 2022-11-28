from anonygraph.assertions import clusters_assertions as cassert
from anonygraph.algorithms.fake_entity_manager import FakeEntityManager
import sys
import itertools
from typing import Dict, List, Set, Tuple, TypeVar
from sortedcontainers import SortedList, SortedDict
import random
import numpy as np
import math
import logging
from time import time

from anonygraph.info_loss import info
from .base_enforcer import BaseEnforcer
from anonygraph.algorithms import PairDistance
from .merge_split_assignment_enforcer import calculate_real_max_dist, remove_invalid_size_and_signature_clusters, assign_valid_clusters, get_cluster_status_str, get_clusters_freq_stats, build_valid_signatures_clusters_for_updation
from anonygraph.algorithms import Cluster, pair_distance
from anonygraph.algorithms.clustering.enforcers import GreedySplitEnforcer
from anonygraph.algorithms.clustering.enforcers.greedy_split_enforcer import find_valid_and_invalid_clusters, count_num_entities, split_big_clusters

logger = logging.getLogger(__name__)

def icde_assign_valid_cluster_or_add_fake_entities(
    valid_clusters: List[Cluster], invalid_clusters: List[Cluster],
    pair_distance: PairDistance, entity2svals: Dict[int, Set[int]], all_sval_ids: Set[int],
    fake_entity_manager: FakeEntityManager, min_size: int,
    min_signature_size: int, max_dist: float
):
    logger.info("assign_valid_cluster_or_add_fake_entities")
    logger.debug("valid_clusters: {}".format(valid_clusters))
    logger.debug("invalid_clusters: {}".format(invalid_clusters))

    num_added_fake_entities = 0
    num_removed_entities = 0
    num_assigned_entities = 0
    num_removed_invalid_clusters = 0
    num_added_fake_entities_clusters = 0

    # all_sval_ids = set()
    # for entity_sval_ids in entity2svals.values():
    #     all_sval_ids.update(entity_sval_ids)

    logger.debug("all_sval_ids: {}".format(all_sval_ids))

    for invalid_cluster in invalid_clusters:
        logger.debug("invalid_cluster: {}".format(invalid_cluster))
        # logger.debug(type(invalid_cluster))
        # raise Exception()
        # count the number of users that can be merged to a valid cluster
        entity2cluster, removed_entity_ids = find_valid_cluster_assignment(
            invalid_cluster, valid_clusters, pair_distance, max_dist
        )

        logger.debug("entity2cluster: {}".format(entity2cluster))
        logger.debug("removed_entity_ids: {}".format(removed_entity_ids))
        # count the number of required fake users

        sval2freq, num_fake_entities = find_num_fake_entities(
            invalid_cluster, entity2svals, fake_entity_manager, min_size, min_signature_size,
            all_sval_ids
        )

        logger.debug("num of removed entities: {} - num of fake entities: {}".format(len(removed_entity_ids), num_fake_entities))
        # if the number of users cannot be merged > the number of fake users: add fake users
        # else: remove cluster
        if (len(removed_entity_ids) > num_fake_entities):
            logger.debug("adding {} fake entities".format(num_fake_entities))
            add_fake_entities(invalid_cluster, sval2freq, fake_entity_manager)
            num_added_fake_entities += num_fake_entities
            num_added_fake_entities_clusters += 1
            valid_clusters.append(invalid_cluster)
        else:
            logger.debug("assigning {} entities to clusters and removing {} entities".format(len(entity2cluster), len(removed_entity_ids)))
            assign_entities_to_valid_clusters(entity2cluster)

            num_removed_entities += len(removed_entity_ids)
            num_assigned_entities += len(entity2cluster)
            num_removed_invalid_clusters += 1

        # if len(entity2cluster) > 0:
        #     raise Exception()

        # if (len(removed_entity_ids) > num_fake_entities):
        #     raise Exception()

    logger.debug("num_added_fake_entities: {}".format(num_added_fake_entities))
    logger.debug("num_removed_entities: {}".format(num_removed_entities))
    logger.debug("num_assigned_entities: {}".format(num_assigned_entities))
    logger.debug("num_added_fake_entities_clusters: {}".format(num_added_fake_entities_clusters))
    logger.debug("num_removed_invalid_clusters: {}".format(num_removed_invalid_clusters))

    logger.debug("valid_clusters (after assign): {}".format(valid_clusters))
    # raise Exception()

class ICDEEnforcer(BaseEnforcer):
    def __init__(self, min_size, max_dist):
        self.min_size = min_size
        self.max_dist = max_dist
        self.gs_enforcer_fn = GreedySplitEnforcer(self.min_size, 1, self.max_dist)

    def __call__(
        self, clusters, pair_distance, entity2svals, all_sval_ids, fake_entity_manager
    ):
        real_max_dist = calculate_real_max_dist(pair_distance, self.max_dist)
        logger.debug("real max dist: {}".format(real_max_dist))

        valid_clusters, invalid_clusters = find_valid_and_invalid_clusters(
            clusters, entity2svals, self.min_size, self.min_signature_size
        )


        num_entities_in_valid_clusters = count_num_entities(valid_clusters)
        num_entities_in_invalid_clusters = count_num_entities(invalid_clusters)

        logger.debug("valid_clusters: {}".format(valid_clusters))
        logger.debug("invalid_clusters: {}".format(invalid_clusters))

        icde_assign_valid_cluster_or_add_fake_entities(
            valid_clusters, invalid_clusters, pair_distance, entity2svals, all_sval_ids,
            fake_entity_manager, self.min_size, self.min_signature_size,
            real_max_dist
        )

        logger.debug("init: num entities in valid clusters: {} - num_entities_in_invalid_clusters: {}".format(num_entities_in_valid_clusters, num_entities_in_invalid_clusters))
        logger.debug("after assign&add: num entities in valid clusters: {}".format(count_num_entities(valid_clusters)))

        # raise Exception()
        cassert.test_invalid_signature_size_clusters(valid_clusters, entity2svals, fake_entity_manager, self.min_signature_size, "after add&assign")
        cassert.test_invalid_min_size_clusters(valid_clusters, self.min_size, "after add&assign")


        new_clusters = split_big_clusters(
            valid_clusters, pair_distance, entity2svals, fake_entity_manager,
            self.min_size, self.min_signature_size
        )

        cassert.test_invalid_signature_size_clusters(new_clusters,entity2svals, fake_entity_manager, self.min_signature_size, "after split")
        cassert.test_invalid_min_size_clusters(new_clusters, self.min_size, "after split")

        return new_clusters

    def update(
        self, current_entity_ids_in_group, removed_entity_ids_in_group, current_real_entity_ids_in_group, current_fake_entity_ids_in_group, pair_distance, entity2svals, fake_entity_manager,
        history
    ):
        # randomly remove users to have at least k users are removed
        removing_entity_ids_in_group = set()
        num_removing_entities = len(removed_entity_ids_in_group)
        num_removing_fake_entities = 0

        fake_entity_ids = list(current_fake_entity_ids_in_group.copy())

        while (num_removing_entities + num_removing_fake_entities < self.min_size):
            # if there is fake user, randomly select a fake user
            if len(fake_entity_ids) > 0:
                fake_entity_idx = random.randint(0, len(fake_entity_ids))
                removing_entity_ids_in_group.add(fake_entity_ids[fake_entity_idx])
                del fake_entity_ids[fake_entity_idx]
            else:
            # randomly select a real user if there is no fake user
                pass

