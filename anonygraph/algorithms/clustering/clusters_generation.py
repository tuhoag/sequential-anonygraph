def warn(*args, **kwargs):
    pass

from io import DEFAULT_BUFFER_SIZE
from anonygraph.algorithms.clustering.calgos.base_clustering_algo import BaseClusteringAlgo
from anonygraph.algorithms.pair_distance import PairDistance
from anonygraph.algorithms.fake_entity_manager import FakeEntityManager
from typing import Dict, Set
from anonygraph.algorithms.clustering.enforcers.greedy_split_enforcer import analyze_unsplitable_clusters, get_signature, get_sval2freq, get_sval2entities
from anonygraph.algorithms.clustering.enforcers import BaseEnforcer
import warnings
warnings.warn = warn

import random
from anonygraph.algorithms.cluster import Cluster
import numpy as np
import sys
import logging
import sklearn.cluster as calgo

import anonygraph.algorithms as algo
from anonygraph.algorithms.clustering.calgos import get_clustering_algorithm
from anonygraph.algorithms.clustering.enforcers import get_enforcer
import anonygraph.info_loss.info as info
import anonygraph.utils.general as utils
from anonygraph.constants import *
from anonygraph.assertions import clusters_assertions as cassert

logger = logging.getLogger(__name__)


def group_entities_by_history(entity_ids,
                              history,
                              num_workers=2) -> Dict[str, Set[int]]:
    # build key for w-1 time instances info
    # group entities by this key
    logger.debug("entities: {} entities".format(len(entity_ids)))
    # if history is None:
    #     return [(None, entity_ids)]

    sub_entity_ids = utils.split_data_to_parts(entity_ids, num_workers)

    data = [(entity_ids, history) for entity_ids in sub_entity_ids]

    mapreduce = algo.MapReduce(
        history_to_keys, group_entities_by_history_key, num_workers=num_workers
    )
    results = mapreduce(data)

    return results


def history_to_keys(inputs):
    logger.debug(inputs)
    entity_ids, history = inputs

    output = []
    for entity_id in entity_ids:
        history_key = history.get_history_key(entity_id)

        output.append((history_key, entity_id))

    logger.debug("inputs: {}".format(inputs))
    logger.debug("output: {}".format(output))
    return output


def group_entities_by_history_key(key_entity_groups):
    key, entity_ids = key_entity_groups
    return (key, set(entity_ids))


def find_big_clusters(clusters, max_size):
    result = []

    for cluster in clusters:
        if len(cluster) > max_size:
            result.append(cluster)

    return result





def is_invalid_size(cluster, min_size):
    return len(cluster) < min_size


def is_invalid_signature_size(cluster, entity2svals, min_signature_size):
    signature = get_signature(cluster, entity2svals)

    return len(signature) < min_signature_size


def is_changed_signature(cluster, entity2svals, history):
    previous_signature = history.get_signature(random.sample(cluster, 1))
    signature = get_signature(cluster, entity2svals)

    return signature != previous_signature


def find_valid_and_invalid_size_and_signature_cluster(
    clusters, entity2svals, min_size, history
):
    invalid_clusters = []
    valid_clusters = []

    for cluster in clusters:
        if is_changed_signature(cluster, entity2svals,
                                history) or is_invalid_size(cluster, min_size):
            invalid_clusters.append(cluster)
        else:
            valid_clusters.append(cluster)

    return valid_clusters, invalid_clusters


def handle_group_of_new_entities(
    group, group_key, pair_distance: PairDistance,
    entity2svals: Dict[int, Set[int]],
    all_sval_ids: Set[int],
    fake_entity_manager: FakeEntityManager,
    min_size: int, min_signature_size: int, calgo_fn: BaseClusteringAlgo,
    enforcer_fn: BaseEnforcer
):
    logger.debug("handle_group_of_new_entities")
    logger.debug(
        'clustering group {} having {} entities'.format(group_key, len(group))
    )
    logger.debug("group: {}".format(group))
    entity_ids = list(group)

    if len(group) >= 2 * min_size:
        logger.debug("clustering")
        clusters = calgo_fn.run(entity_ids, pair_distance)
        logger.debug("num clusters after clustering {}".format(len(clusters)))

    elif len(group) > 0:
        logger.debug("creating single cluster")
        clusters = [Cluster.from_iter(group)]
    else:
        # there is no new user
        return []

    # modify the clusters
    clusters = enforcer_fn(
        clusters, pair_distance, entity2svals, all_sval_ids, fake_entity_manager
    )

    logger.info('generated clusters: {}'.format(clusters))
    return clusters

def calculate_num_removable_entities_without_being_invalid_size_signature(entity_ids_set, entity2svals, fake_entity_manager, min_size):
    sval2entities = get_sval2entities(entity_ids_set, entity2svals, fake_entity_manager)

    num_removable_entities_signature = 0
    for sval_entities in sval2entities.values():
        num_removable_entities_signature += len(sval_entities) - 1
        logger.debug("sval_entitites: {}".format(sval_entities))
        logger.debug("num_removable_entities_signature: {}".format(num_removable_entities_signature))

    num_removable_entities_size = max(0, len(entity_ids_set) - min_size)
    num_removable_entities = min(num_removable_entities_signature, num_removable_entities_size)

    logger.debug("entities: {}".format(entity_ids_set))
    logger.debug("num_removable_entities_size: {}".format(num_removable_entities_size))
    logger.debug("num_removable_entities_signature: {}".format(num_removable_entities_signature))
    logger.debug("num_removable_entities: {}".format(num_removable_entities))

    return num_removable_entities


def handle_group_of_existed_entities(
    group, group_key, current_real_entity_ids, pair_distance, entity2svals,
    fake_entity_manager, history, min_size, min_signature_size, calgo_fn,
    enforcer_fn
):
    logger.debug("handle_group_of_existed_entities")
    logger.debug("group: {}".format(group))

    # find users that are in the current subgraph
    current_real_entity_ids_in_group = current_real_entity_ids.intersection(group)
    logger.debug(
        "current_real_entity_ids_in_group (len: {}): {}".format(len(current_real_entity_ids_in_group), current_real_entity_ids_in_group
    ))

    logger.debug("fake_entity_manager: {}".format(fake_entity_manager))
    current_fake_entity_ids_in_group = fake_entity_manager.get_fake_entity_ids_in_entities(
        group
    )
    logger.debug(
        "current_fake_entity_ids_in_group (len: {}): {}".format(len(current_fake_entity_ids_in_group), current_fake_entity_ids_in_group)
    )

    current_entity_ids_in_group = current_real_entity_ids_in_group.union(current_fake_entity_ids_in_group)
    logger.debug("current_entity_ids_in_group: {}".format(current_entity_ids_in_group))

    # find removed entities
    removed_entity_ids_in_group = group.difference(current_entity_ids_in_group)
    logger.debug("removed_entity_ids_in_group (len: {}): {}".format(len(removed_entity_ids_in_group), removed_entity_ids_in_group))

    # remove users such that amount of invalid signatures are minimized
    clusters = enforcer_fn.update(current_entity_ids_in_group, removed_entity_ids_in_group, current_real_entity_ids_in_group, current_fake_entity_ids_in_group, pair_distance, entity2svals, fake_entity_manager, history)

    return clusters

class ClustersGeneration(object):
    def __init__(
        self, min_size, window_size, min_signature_size, calgo_name,
        enforcer_name, args
    ):
        self.min_size = min_size
        self.window_size = window_size
        self.min_signature_size = min_signature_size
        self.calgo_fn = get_clustering_algorithm(calgo_name, args)
        self.enforcer_fn = get_enforcer(enforcer_name, args)
        self.args = args

    def run(
        self, pair_distance, entity2svals, all_sval_ids, history, fake_entity_manager,
        time_inst
    ):
        # logger.info(type(entity2sensitive_vals))
        # raise Exception()
        logger.debug(history)
        if self.window_size != -1:
            num_removals = history.num_time_instances - self.window_size + 1
            logger.debug('num removal: {}'.format(num_removals))
            for _ in range(num_removals):
                history.remove_first_history()

        current_real_entity_ids = pair_distance.entity_ids_set
        logger.debug(
            "current entity ids ({}): {}".format(
                len(current_real_entity_ids), current_real_entity_ids
            )
        )

        released_entity_ids = history.entity_ids
        logger.debug(
            "released entity ids ({}): {}".format(
                len(released_entity_ids), released_entity_ids
            )
        )

        all_entity_ids = current_real_entity_ids.union(released_entity_ids)
        logger.debug(
            "all entity ids ({}): {}".format(
                len(all_entity_ids), all_entity_ids
            )
        )

        logger.debug(
            "num entities: current: {} - released: {} - all: {}".format(
                len(current_real_entity_ids), len(released_entity_ids),
                len(all_entity_ids)
            )
        )

        # raise Exception()
        groups = group_entities_by_history(list(all_entity_ids), history)
        logger.debug("group keys: {} groups".format(len(groups)))
        logger.debug('groups: {}'.format(list(map(lambda item: item[1], groups))))

        # if num_removals > 0:
        # raise Exception()
        clusters = algo.Clusters()

        for group_key, group in groups:
            logger.debug(
                "group: {} with {} entities (new: {}).".format(
                    group_key, len(group), group_key == EMPTY_HISTORY_KEY
                )
            )

            if group_key == EMPTY_HISTORY_KEY:
                current_clusters = handle_group_of_new_entities(
                    group, group_key, pair_distance, entity2svals, all_sval_ids,
                    fake_entity_manager, self.min_size, self.min_signature_size,
                    self.calgo_fn, self.enforcer_fn
                )
                clusters.add_clusters(current_clusters)
                cassert.test_invalid_min_size_clusters(clusters, self.min_size, "insert")
            else:
                current_clusters = handle_group_of_existed_entities(
                    group, group_key, current_real_entity_ids, pair_distance,
                    entity2svals, fake_entity_manager, history, self.min_size,
                    self.min_signature_size, self.calgo_fn, self.enforcer_fn
                )

                clusters.add_clusters(current_clusters)

        logger.info("clusters after clustering: {}".format(clusters))
        # if len(clusters) == 0:
        #     raise Exception()
        return clusters
