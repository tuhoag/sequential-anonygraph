from token import ELLIPSIS
from anonygraph.assertions import clusters_assertions as cassert
from math import pi
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

logger = logging.getLogger(__name__)


def count_num_entities(clusters):
    count = 0
    for cluster in clusters:
        count += len(cluster)

    return count

class GreedySplitEnforcer(BaseEnforcer):
    def __init__(self, min_size, min_signature_size, max_dist):
        self.min_size = min_size
        self.min_signature_size = min_signature_size
        self.max_dist = max_dist

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

        assign_valid_cluster_or_add_fake_entities(
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

        sign2entities = group_by_signatures(current_entity_ids_in_group, history)
        logger.debug("init sign2entities: {}".format(sign2entities))

        valid_sign2entities, entities_in_invalid_clusters = delete_invalid_cluster(sign2entities, entity2svals, fake_entity_manager, self.min_size, self.min_signature_size)
        logger.debug("init valid_sign2entities (len: {}): {}".format(len(valid_sign2entities), valid_sign2entities))
        logger.debug("entities_in_invalid_clusters: (len: {}): {}".format(len(entities_in_invalid_clusters), entities_in_invalid_clusters))

        valid_current_entity_ids = set()
        for entity_ids in valid_sign2entities.values():
            valid_current_entity_ids.update(entity_ids)

        # remove at least min_size entities to protect the deleted entities
        removed_entity_ids_in_group.update(entities_in_invalid_clusters)
        logger.debug("updated removed_entity_ids_in_group (len: {}): {}".format(len(removed_entity_ids_in_group), removed_entity_ids_in_group))

        removing_real_entity_ids_in_group, removing_fake_entity_ids_in_group = delete_entities_to_protect_deleted_ones(
            valid_sign2entities, current_fake_entity_ids_in_group, current_real_entity_ids_in_group, removed_entity_ids_in_group, entity2svals, fake_entity_manager, self.min_size, self.min_signature_size
        )

        logger.debug("removing_real_entity_ids_in_group: {}".format(removing_real_entity_ids_in_group))
        logger.debug("removing_fake_entity_ids_in_group: {}".format(removing_fake_entity_ids_in_group))


        clusters = split_clusters_with_same_signature(valid_sign2entities, pair_distance, entity2svals, fake_entity_manager, self.min_size, self.min_signature_size)


        # do some tests
        # test min_size & min_signature_size
        cassert.test_invalid_min_size_clusters(clusters, self.min_size, "existed_group")
        cassert.test_invalid_signature_size_clusters(clusters, entity2svals, fake_entity_manager, self.min_signature_size, "existed_group")

        # test same signature
        cassert.test_invalid_same_signature_clusters(clusters, entity2svals, fake_entity_manager, history)


        # test deleted users
        cassert.test_min_size_deleted_entities(clusters, removed_entity_ids_in_group, valid_current_entity_ids, self.min_size)
        cassert.test_removed_and_removing_entities_not_in_clusters(clusters, removed_entity_ids_in_group, valid_current_entity_ids)

        return clusters


def group_by_signatures(entity_ids, history):
    sign2entities = {}
    for entity_id in entity_ids:
        signature_key = frozenset(history.get_signature(entity_id))
        logger.debug("entity_id: {} - signature_key: {}".format(entity_id, signature_key))
        current_entity_ids = sign2entities.get(signature_key, None)
        if current_entity_ids is None:
            current_entity_ids = set()
            sign2entities[signature_key] = current_entity_ids

        current_entity_ids.add(entity_id)

    return sign2entities

def delete_invalid_cluster(sign2entities, entity2svals, fake_entity_manager, min_size, min_signature_size):
    invalid_clusters = []
    entities_in_invalid_clusters = set()
    valid_sign2entities = {}

    for signature_key, entities in sign2entities.items():
        current_signature = get_signature(entities, entity2svals, fake_entity_manager)

        logger.debug("signature: {}".format(signature_key))
        logger.debug("entities: {}".format(entities))
        logger.debug("current signature: {}".format(current_signature))

        if signature_key != current_signature or len(entities) < min_size or len(current_signature) < min_signature_size:
            entities_in_invalid_clusters.update(entities)

            logger.debug("cluster: {} is invalid (signature: {} - size: {} - signature_size: {})".format(entities, signature_key != current_signature, len(entities) < min_size, len(current_signature) < min_signature_size))
        else:
            valid_sign2entities[signature_key] = entities

    return valid_sign2entities, entities_in_invalid_clusters

def delete_entities_to_protect_deleted_ones(
    valid_sign2entities, current_fake_entity_ids_in_group, current_real_entity_ids_in_group, removed_entity_ids_in_group, entity2svals, fake_entity_manager, min_size, min_signature_size
):
    logger.debug("init valid_sign2entities: {}".format(valid_sign2entities))
    logger.debug("init current_fake_entity_ids_in_group: {}".format(current_fake_entity_ids_in_group))
    logger.debug("init current_real_entity_ids_in_group: {}".format(current_real_entity_ids_in_group))
    # num_entities = len(current_fake_entity_ids_in_group) + len(current_real_entity_ids_in_group)
    num_entities = sum(map(lambda entities: len(entities), valid_sign2entities.values()))
    # assert num_entities == num_entities2, "Num of entities in valid_sign2entities (len: {}) != that of those in fake and real set (len: {})".format(num_entities2, num_entities)
    logger.debug("valid_sign2entities (len: {}): {}".format(num_entities, valid_sign2entities))

    if len(removed_entity_ids_in_group) >= min_size or len(removed_entity_ids_in_group) == 0:
        # they are protected, do not need to remove any entity
        return set(), set()

    num_removing_entities = min_size - len(removed_entity_ids_in_group)

    if num_removing_entities >= num_entities:
        # remove all entities in the group
        return current_real_entity_ids_in_group, current_fake_entity_ids_in_group


    num_removable_entities = calculate_num_removable_entities(valid_sign2entities, min_size)
    logger.debug("init num_removable_entities: {} - num_removing_entities: {}".format(num_removable_entities, num_removing_entities))

    removing_fake_entity_ids = delete_while_keeping_balance_and_valid(valid_sign2entities, current_fake_entity_ids_in_group, num_removing_entities, entity2svals, fake_entity_manager, min_size)
    num_removing_fake_entities = len(removing_fake_entity_ids)
    logger.debug("removed_fake_entity_ids: {}".format(removing_fake_entity_ids))
    logger.debug("valid_sign2entities: {}".format(valid_sign2entities))

    removing_real_entity_ids = delete_while_keeping_balance_and_valid(valid_sign2entities, current_real_entity_ids_in_group, num_removing_entities - num_removing_fake_entities, entity2svals, fake_entity_manager, min_size)
    num_removing_real_entities = len(removing_real_entity_ids)
    logger.debug("removing_real_entity_ids: {}".format(removing_real_entity_ids))
    logger.debug("valid_sign2entities: {}".format(valid_sign2entities))


    remaining_removing_entity_ids = delete_entities_based_on_cost(valid_sign2entities, current_fake_entity_ids_in_group, current_real_entity_ids_in_group, num_removing_entities - num_removing_fake_entities - num_removing_real_entities, entity2svals, fake_entity_manager, min_size)
    num_removing_remaining_entities = len(remaining_removing_entity_ids)
    logger.debug("remaining_removing_entity_ids: {}".format(remaining_removing_entity_ids))
    logger.debug("valid_sign2entities: {}".format(valid_sign2entities))

    new_num_entities = sum(map(lambda item: len(item), valid_sign2entities.values()))
    logger.debug("valid_sign2entities (len: {}): {}".format(new_num_entities, valid_sign2entities))

    logger.debug("num_removing_real_entities: {} - num_removing_fake_entities: {} - num_removing_entities: {} - num_removing_remaining_entities: {}".format(num_removing_real_entities, num_removing_fake_entities, num_removing_remaining_entities, num_removing_entities))
    if num_removing_fake_entities + num_removing_remaining_entities + num_removing_real_entities < num_removing_entities:
        logger.debug("num_removing_fake_entities ({}) + num_removing_real_entities ({}) + num_removing_remaining_entities ({}) != num_removing_entities ({})".format(num_removing_fake_entities, num_removing_real_entities, num_removing_remaining_entities, num_removing_entities))

        raise Exception("num_removing_fake_entities ({}) + num_removing_real_entities ({}) + num_removing_remaining_entities ({}) != num_removing_entities ({})".format(num_removing_fake_entities, num_removing_real_entities, num_removing_remaining_entities, num_removing_entities))


    assert num_entities - new_num_entities == num_removing_remaining_entities + num_removing_fake_entities + num_removing_real_entities, "The updated valid_sign2entities's len ({}) is different num of entities than the extracted removing ids (remaining: {} + fake: {} + real: {} = {}).".format(num_entities - new_num_entities, num_removing_remaining_entities, num_removing_fake_entities, num_removing_real_entities, num_removing_fake_entities + num_removing_remaining_entities + num_removing_real_entities)

    num_removed_entities_in_group = len(removed_entity_ids_in_group)
    assert num_removing_fake_entities + num_removing_remaining_entities + num_removing_real_entities + num_removed_entities_in_group >= min_size, "Not enough removing entities (fake: {} - real: {} - remaining: {} - original removed: {})".format(num_removing_fake_entities, num_removing_real_entities, num_removing_remaining_entities, num_removed_entities_in_group)

    logger.debug("min_size: {} - num_removed_entities_in_group: {}".format(min_size, num_removed_entities_in_group))
    logger.debug("current_real_entity_ids_in_group (len: {}): {}".format(len(current_real_entity_ids_in_group), current_real_entity_ids_in_group))
    logger.debug("current_fake_entity_ids_in_group (len: {}): {}".format(len(current_fake_entity_ids_in_group), current_fake_entity_ids_in_group))

    logger.debug("removing_fake_entity_ids (len: {}): {}".format(num_removing_fake_entities, removing_fake_entity_ids))
    logger.debug("removing_real_entity_ids (len: {}): {}".format(num_removing_real_entities, removing_real_entity_ids))
    logger.debug("removing_remaining_entity_ids (len: {}): {}".format(num_removing_remaining_entities, remaining_removing_entity_ids))

    # raise Exception()
    remaining_removing_fake_entity_ids = remaining_removing_entity_ids.intersection(current_fake_entity_ids_in_group)
    remaining_removing_real_entity_ids = remaining_removing_entity_ids.intersection(current_real_entity_ids_in_group)
    logger.debug("remaining_removing_fake_entity_ids (len: {}): {}".format(len(remaining_removing_fake_entity_ids), remaining_removing_fake_entity_ids))
    logger.debug("remaining_removing_real_entity_ids (len: {}): {}".format(len(remaining_removing_real_entity_ids), remaining_removing_real_entity_ids))

    return removing_real_entity_ids.union(remaining_removing_real_entity_ids), removing_fake_entity_ids.union(remaining_removing_fake_entity_ids)


def calculate_num_removable_entities(valid_sign2entities, min_size):
    num_removable_entities = 0
    for entity_ids_set in valid_sign2entities.values():
        num_removable_entities += max(0, len(entity_ids_set) - min_size)

    return num_removable_entities


def sort_clusters_by_removal_cost(valid_sign2entities, current_fake_entity_ids_in_group, current_real_entity_ids_in_group):
    sorted_clusters = SortedList(key=lambda item: item[0])

    for signature_key, entity_ids_set in valid_sign2entities.items():
        num_fake_entities = len(entity_ids_set.intersection(current_fake_entity_ids_in_group))
        num_real_entities = len(entity_ids_set.intersection(current_real_entity_ids_in_group))

        cluster_cost = num_real_entities - num_fake_entities
        logger.debug("cluster: {} - real: {} - fake: {} - cost: {}".format(entity_ids_set, num_real_entities, num_fake_entities, cluster_cost))
        sorted_clusters.add((cluster_cost, signature_key, entity_ids_set))

    return sorted_clusters

def delete_entities_based_on_cost(valid_sign2entities, current_fake_entity_ids_in_group, current_real_entity_ids_in_group, num_removing_entities, entity2svals, fake_entity_manager, min_size):
    if num_removing_entities <= 0:
        return set()

    logger.debug("init num_removing_entities: {}".format(num_removing_entities))

    sorted_entity_ids_set = sort_clusters_by_removal_cost(valid_sign2entities, current_fake_entity_ids_in_group, current_real_entity_ids_in_group)
    logger.debug("sorted_clusters: {}".format(sorted_entity_ids_set))

    remaining_removing_entity_ids = set()

    # while num_removing_entities > 0:
    #   select the smallest cost entity_ids_set
    #   remove the selected entities
    #   update the num_removing_entities
    while(num_removing_entities > 0):
        cost, signature_key, entity_ids_set = sorted_entity_ids_set.pop(0)
        logger.debug("entity_ids_set: {} - cost: {}".format(entity_ids_set, cost))

        # update num removing entities
        logger.debug("num_removing_entities: {}".format(num_removing_entities))

        remaining_removing_entity_ids.update(entity_ids_set)

        logger.debug("remaining_removing_entity_ids: {}".format(remaining_removing_entity_ids))

        # remove all entities
        del valid_sign2entities[signature_key]
        num_removing_entities = num_removing_entities - len(entity_ids_set)

    # if len(sorted_entity_ids_set) > 1:
    #     raise Exception()

    return remaining_removing_entity_ids



def delete_while_keeping_balance_and_valid(valid_sign2entities, removable_entity_ids, num_removing_entities, entity2svals, fake_entity_manager, min_size):
    removed_entity_ids = set()

    logger.debug("valid_sign2entities: {}".format(valid_sign2entities))
    logger.debug("removable_entity_ids: {}".format(removable_entity_ids))
    logger.debug("num_removing_entities: {}".format(num_removing_entities))
    logger.debug("min_size: {}".format(min_size))

    if num_removing_entities <= 0:
        return removed_entity_ids

    for signature_key, entity_ids_set in valid_sign2entities.items():
        logger.debug("entity_ids_set: {}".format(entity_ids_set))
        removing_entity_ids_in_cluster = set()

        if len(entity_ids_set) == min_size:
            logger.debug("skip cluster since size ({}) == min_size ({})".format(len(entity_ids_set), min_size))
            continue
        elif len(entity_ids_set) < min_size:
            raise Exception("cluster: {} has less than min_size ({}) entities".format(entity_ids_set, min_size))

        # num of removable entities without making the cluster vioalte min_size constraint
        num_removable_entities = max(0, len(entity_ids_set) - min_size)
        logger.debug("init num_removable_entities_in_cluster: {}".format(num_removable_entities))

        removable_entity_ids_in_cluster = removable_entity_ids.intersection(entity_ids_set)
        logger.debug("removable_entity_ids_in_cluster: {}".format(removable_entity_ids_in_cluster))

        if len(removable_entity_ids_in_cluster) == 0:
            logger.debug("skip cluster since there is no entities to be deleted")
            continue

        sval2entities = get_sval2entities(entity_ids_set, entity2svals, fake_entity_manager)
        sorted_sval2freq = SortedList(key=lambda item: -item[1],
            iterable=map(lambda item: (item[0], len(item[1])), sval2entities.items())
        )

        logger.debug("sval2entities: {}".format(sval2entities))
        logger.debug("sorted_sval2freq: {}".format(sorted_sval2freq))

        for sval_id, sval_freq in sorted_sval2freq:
            logger.debug("sval_id: {} - sval_freq: {}".format(sval_id, sval_freq))

            sval_entity_ids_set = sval2entities[sval_id]
            sval_removable_entity_ids_set = sval_entity_ids_set.intersection(removable_entity_ids_in_cluster)

            logger.debug("sval_entity_ids_set: {}".format(sval_entity_ids_set))
            logger.debug("sval_removable_entity_ids_set: {}".format(sval_removable_entity_ids_set))

            if len(sval_removable_entity_ids_set) <= num_removing_entities:
                # num_sval_removing_entities = len(sval_removable_entity_ids_set) - num_removing_entities
                num_sval_removing_entities = len(sval_removable_entity_ids_set)
            else:
                num_sval_removing_entities = num_removing_entities

            logger.debug("num_sval_removing_entities: {}".format(num_sval_removing_entities))

            # check if removing entities make this sval has no entitites
            if num_sval_removing_entities == sval_freq:
                num_sval_removing_entities = num_sval_removing_entities - 1

            # check for the valid size of the whole cluster after removing
            if num_sval_removing_entities >= num_removable_entities:
                num_sval_removing_entities = num_removable_entities
                num_removable_entities = 0
            else:
                num_removable_entities = num_removable_entities - num_sval_removing_entities

            if num_sval_removing_entities == 0:
                continue

            removing_entity_ids_in_cluster.update(random.sample(sval_removable_entity_ids_set, k=num_sval_removing_entities))
            logger.debug("removed_entity_ids_in_cluster: {}".format(removed_entity_ids))

            num_removing_entities = num_removing_entities - num_sval_removing_entities
            logger.debug("num_removing_entities: {}".format(num_removing_entities))

            # assert len(sval_entity_ids_set) > 0, "Cannot remove all entities in sval_id: {}".format(sval_id)
            assert len(entity_ids_set.difference(removing_entity_ids_in_cluster)) >= min_size, "Cluster: {} must has at least min_size: {} entities after removing {}".format(entity_ids_set, min_size, removing_entity_ids_in_cluster)

            # check for valid signature
            assert len(sval_entity_ids_set.difference(removing_entity_ids_in_cluster)) > 0, "Sval_id: {} ({}) must have at least 1 entities that are not inremoving_entity_ids_in_cluster".format(sval_id, sval_entity_ids_set, removing_entity_ids_in_cluster)

        removed_entity_ids.update(removing_entity_ids_in_cluster)
        entity_ids_set.difference_update(removing_entity_ids_in_cluster)
        logger.debug("removed_entity_ids: {}".format(removed_entity_ids))
        logger.debug("entity_ids_set: {}".format(entity_ids_set))

        assert len(entity_ids_set) >= min_size, "Cluster: {} must has at least min_size ({}) entities".format(entity_ids_set, min_size)


    logger.debug("valid_sign2entities: {}".format(valid_sign2entities))
    logger.debug("removed_entity_ids: {}".format(removed_entity_ids))

    return removed_entity_ids


def split_a_cluster_with_same_signature(entity_ids_set, pair_distance, entity2svals, fake_entity_manager, min_size):
    logger.debug("init cluster: {}".format(entity_ids_set))
    if len(entity_ids_set) < min_size * 2:
        return [Cluster.from_iter(entity_ids_set)]

    # find smallest sval whose freq is smallest
    sval2entities = get_sval2entities(entity_ids_set, entity2svals, fake_entity_manager)
    logger.debug("init sval2entities: {}".format(sval2entities))
    # sval2freq = get_sval2freq(entity_ids_set, entity2svals, fake_entity_manager)
    smallest_sval = (None, sys.maxsize)
    for sval_id, entities in sval2entities.items():
        num_entities = len(entities)

        if num_entities < smallest_sval[1]:
            smallest_sval = sval_id, num_entities

    # create a cluster for each entity in the found sval
    smallest_sval_entity_ids = sval2entities[smallest_sval[0]]
    num_spliting_clusters_by_size = math.floor(len(entity_ids_set) / min_size)
    num_spliting_clusters = min(len(smallest_sval_entity_ids), num_spliting_clusters_by_size)
    init_selected_entity_ids_for_cluster = random.sample(smallest_sval_entity_ids, k=num_spliting_clusters)
    clusters = [Cluster.from_iter([entity_id]) for entity_id in init_selected_entity_ids_for_cluster]

    logger.debug("smallest_sval_entity_ids: {}".format(smallest_sval_entity_ids))
    logger.debug("init_selected_entity_ids_for_cluster: {}".format(init_selected_entity_ids_for_cluster))
    logger.debug("spliting to {} clusters: {}".format(len(clusters), clusters))

    # for each cluster, for each remaining sval, find the entity that is closesest to the cluster
    if num_spliting_clusters == len(smallest_sval_entity_ids):
        del sval2entities[smallest_sval[0]]
    else:
        smallest_sval_entity_ids.difference_update(init_selected_entity_ids_for_cluster)

    logger.debug("updated sval2entities: {}".format(sval2entities))
    for cluster in clusters:
        for sval_id, sval_entities in sval2entities.items():
            # find closest entity of sval_id in for this cluster
            if sval_id == smallest_sval[0]:
                continue

            closest_entity = (sys.maxsize, None)

            for sval_entity_id in sval_entities:
                cluster_dist = calculate_entity2cluster_distance(sval_entity_id, cluster, pair_distance)

                if cluster_dist < closest_entity[0]:
                    closest_entity = (cluster_dist, sval_entity_id)

            cluster.add_entity(closest_entity[1])
            sval_entities.remove(closest_entity[1])

    logger.debug("base clusters: {}".format(clusters))

    # add to satisfy min_size
    for cluster in clusters:
        num_adding_entities = max(0, min_size - len(cluster))
        closest_entity_ids = SortedList(key=lambda item: item[0])

        logger.debug("cluster: {} - num_adding_entities: {}".format(cluster, num_adding_entities))
        logger.debug("sval2entities: {}".format(sval2entities))

        for sval_id, sval_entities in sval2entities.items():
            for sval_entity_id in sval_entities:
                cluster_dist = calculate_entity2cluster_distance(sval_entity_id, cluster, pair_distance)

                closest_entity_ids.add((cluster_dist, sval_id, sval_entity_id))

        logger.debug("closest_entity_ids: {}".format(closest_entity_ids))

        while(num_adding_entities > 0):
            cluster_dist, sval_id, entity_id = closest_entity_ids.pop(0)
            logger.debug("add {} to cluster: {}".format(entity_id, cluster))
            cluster.add_entity(entity_id)
            sval2entities[sval_id].remove(entity_id)
            # sval_entities.remove(entity_id)
            num_adding_entities = num_adding_entities - 1


    # add the remaining entities to their closest cluster
    for _, sval_entities in sval2entities.items():
        for sval_entity_id in sval_entities:
            closest_cluster_tuple = (sys.maxsize, None)

            for cluster in clusters:

                cluster_dist = calculate_entity2cluster_distance(sval_entity_id, cluster, pair_distance)
                logger.debug("sval_entity_id: {} - cluster_dist: {}".format(sval_entity_id, cluster_dist))
                if cluster_dist < closest_cluster_tuple[0]:
                    closest_cluster_tuple = (cluster_dist, cluster)
                    logger.debug("updated closest_cluster_tuple: {}".format(closest_cluster_tuple))

            closest_cluster_tuple[1].add_entity(sval_entity_id)

    logger.debug("final clusters: {}".format(clusters))

    # raise Exception()
    return clusters


def calculate_entity2cluster_distance(entity_id, cluster_entity_ids, pair_distance):
    cluster_dist = 0
    for cluster_entity_id in cluster_entity_ids:
        cluster_dist = max(pair_distance.get_distance(entity_id, cluster_entity_id), cluster_dist)

    return cluster_dist

def split_clusters_with_same_signature(valid_sign2entities, pair_distance, entity2svals, fake_entity_manager, min_size, min_signature_size):
    logger.debug("init valid_sign2entities: {}".format(valid_sign2entities))
    clusters = []

    unsplitable_clusters = []

    for signature_key, entity_ids_set in valid_sign2entities.items():
        splited_clusters = split_a_cluster_with_same_signature(entity_ids_set, pair_distance, entity2svals, fake_entity_manager, min_size)

        if len(splited_clusters) == 1 and len(splited_clusters[0]) >= min_size * 2:
            unsplitable_clusters.append(splited_clusters[0])


        cassert.test_same_signature_splited_clusters(entity_ids_set, splited_clusters, entity2svals, fake_entity_manager)
        cassert.test_invalid_min_size_clusters(splited_clusters, min_size, "split_clusters")

        clusters.extend(splited_clusters)

    # analyze_unsplitable_clusters(unsplitable_clusters, pair_distance, entity2svals, fake_entity_manager, min_size, min_signature_size)

    return clusters

def find_valid_and_invalid_clusters(
    clusters: List[Cluster], entity2svals: Dict[int, Set[int]], min_size: float,
    min_signature_size: float
) -> Tuple[List[Cluster], List[Cluster]]:
    valid_clusters = []
    invalid_clusters = []

    for cluster in clusters:
        signature = info.get_generalized_signature_info_from_dict(
            entity2svals, None, cluster
        )

        cluster_size = len(cluster)
        signature_size = len(signature)

        logger.debug(
            "cluster size: {} - signature_size: {} - min_size: {} - min_signature_size: {}"
            .format(cluster_size, signature_size, min_size, min_signature_size)
        )

        logger.debug(
            get_cluster_status_str(
                cluster_size, signature_size, min_size, min_signature_size
            )
        )

        if signature_size >= min_signature_size and cluster_size >= min_size:
            valid_clusters.append(cluster)
        else:
            invalid_clusters.append(cluster)

    return valid_clusters, invalid_clusters


def find_valid_cluster_assignment(
    invalid_cluster: Cluster, valid_clusters: List[Cluster],
    pair_distance: PairDistance, max_dist: float
) -> Tuple[Dict[int, Cluster], Set[int]]:
    """Find a valid cluster for entities. An entity can only be merged to a cluster if their distance is less than or equal to the maximum distance.

    Args:
        invalid_cluster (set[int]): Set of entity ids in an invalid cluster
        valid_clusters (list[cluster]): list of Cluster objects
        max_dist (float): maximum distance

    Returns:
        dict: cluster2entities
        set: set of removed entity ids
    """
    entity2cluster = {}
    removed_entity_ids = set()

    for entity_id in invalid_cluster:
        closest_valid_cluster = (max_dist, None)

        for cluster in valid_clusters:
            cluster_dist = -sys.maxsize
            for entity2_id in cluster:
                dist = pair_distance.get_distance(entity_id, entity2_id)

                cluster_dist = max(cluster_dist, dist)

            if cluster_dist <= closest_valid_cluster[0]:
                closest_valid_cluster = (cluster_dist, cluster)

        if closest_valid_cluster[1] is not None:
            # cluster_key = frozenset(closest_valid_cluster[1])
            # valid_cluster = entity2cluster.get(entity_id, None)
            # if valid_cluster is None:
            entity2cluster[entity_id] = closest_valid_cluster[1]

            # entity_ids.add(entity_id)
        else:
            removed_entity_ids.add(entity_id)

    return entity2cluster, removed_entity_ids


def find_num_fake_entities(
    invalid_cluster, entity2svals, fake_entity_manager, min_size, min_signature_size, all_sval_ids
):
    """Find number of fake entities required to make this cluster valid.

    Args:
        invalid_cluster (set(int)): Set of entity ids in an invalid cluster
        entity2svals (dict[int:set[int]]): Dictionary of entity ids and their sval_ids
        min_size (int): Minimum cluster size
        min_signature_size (int): Min signature size

    Returns:
        dict[int:int]: dictionary of sval_ids and their number of fake entities required to be added
        int: number of required fake entities
    """
    # calculate the freq of signatures
    sval2freq = get_sval2freq(invalid_cluster, entity2svals, fake_entity_manager)
    signature_size = len(sval2freq)

    sorted_sval2freq = SortedDict(sval2freq)

    logger.debug("init sval2freq: {}".format(sval2freq))
    logger.debug("init sorted_sval2freq: {}".format(sorted_sval2freq))
    # calculate size
    cluster_size = len(invalid_cluster)

    # if cluster size >= min_size and signature_size >= min_signature_size: return
    if cluster_size >= min_size and signature_size >= min_signature_size:
        return {}, 0

    sval2fake_freq = {}
    num_fake_entities = 0
    # calculate the number of required for each svals
    # if signature_size < min_signature_size:
    #   randomly select missing min_signature_size - signature_size svals.
    #   add a fake entities for each selected svals
    missing_sval_ids = all_sval_ids.difference(set(sval2freq.keys()))
    logger.debug("init missing_sval_ids: {}".format(missing_sval_ids))
    logger.debug("all_sval_ids: {}".format(all_sval_ids))
    logger.debug("min_signature_size: {}".format(min_signature_size))
    while (signature_size < min_signature_size):
        if len(missing_sval_ids) == 0:
            raise Exception(
                "num of svals ({}) is less than min_signature_size ({})".format(
                    len(all_sval_ids), min_signature_size
                )
            )

        sval_id = random.sample(missing_sval_ids, 1)[0]
        missing_sval_ids.remove(sval_id)

        num_fake_entities += 1
        signature_size += 1
        cluster_size += 1

        sorted_sval2freq[sval_id] = sorted_sval2freq.get(sval_id, 0) + 1
        sval2fake_freq[sval_id] = sval2fake_freq.get(sval_id, 0) + 1

    # if cluster_size + added fake entities < min_size
    #   in each iteration, select sval that have least entities to add fake entities
    # sorted_sval_freq = SortedList(key=lambda item: item[0])

    logger.debug("after adding for l: sval2fake_freq: {}".format(sval2fake_freq))
    logger.debug("after adding for l: sorted_sval2freq: {}".format(sorted_sval2freq))
    logger.debug("after adding for l: cluster_size: {} - signature_size: {}".format(cluster_size, signature_size))

    while (cluster_size < min_size):
        sval_id, freq = sorted_sval2freq.popitem(0)

        sval2fake_freq[sval_id] = sval2fake_freq.get(sval_id, 0) + 1
        sorted_sval2freq[sval_id] = freq + 1

        cluster_size += 1
        num_fake_entities += 1

        logger.debug("adding for k: sval2fake_freq: {}".format(sval2fake_freq))
        logger.debug("adding for k: sorted_sval2freq: {}".format(sorted_sval2freq))

    logger.debug("after adding for k: sval2fake_freq: {}".format(sval2fake_freq))
    logger.debug("after adding for k: sorted_sval2freq: {}".format(sorted_sval2freq))
    logger.debug("after adding for k: cluster_size: {} - signature_size: {}".format(cluster_size, signature_size))

    return sval2fake_freq, num_fake_entities


def add_fake_entities(invalid_cluster, sval2count, fake_entity_manager):
    count = 0
    for sval_id, freq in sval2count.items():
        for _ in range(freq):
            fake_entity_id = fake_entity_manager.create_new_fake_entity(sval_id)
            invalid_cluster.add_entity(fake_entity_id)
            count += 1

    logger.debug("added {} fake entities".format(count))


def assign_entities_to_valid_clusters(entity2cluster):
    count = 0

    for entity_id, cluster in entity2cluster.items():
        cluster.add_entity(entity_id)
        count += 1

    logger.debug("assigned {} entities to their nearest valid clusters".format(count))
    # raise Exception()


def assign_valid_cluster_or_add_fake_entities(
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

def split_big_clusters(
    clusters, pair_distance, entity2svals, fake_entity_manager, min_size,
    min_signature_size
):
    new_clusters = []
    unsplitable_clusters = []
    # count_dict = {}
    # for cluster in clusters:
    #     count_dict[str(type(cluster))] = count_dict.get(str(type(cluster)), 0) + 1

    # logger.debug(count_dict)
    # raise Exception()
    open_clusters = [cluster for cluster in clusters]

    while len(open_clusters) > 0:
        cluster = open_clusters.pop(0)

        signature = info.get_generalized_signature_info_from_dict(
            entity2svals, fake_entity_manager, cluster
        )

        cluster_size = len(cluster)
        signature_size = len(signature)

        logger.debug(
            "current cluster has {} entities (remaining clusters: {})".format(
                len(cluster), len(open_clusters)
            )
        )
        logger.debug(
            get_cluster_status_str(
                cluster_size, signature_size, min_size, min_signature_size
            )
        )

        if cluster_size < min_size or signature_size < min_signature_size:
            raise Exception(
                "cluster (size: {} - sig size: {}): {} is invalid (min_size: {} - min_sig_size: {}"
                .format(
                    cluster_size, signature_size, cluster, min_size,
                    min_signature_size
                )
            )

        # current_clusters = [cluster]

        if cluster_size < 2 * min_size:
            logger.debug("add cluster")
            new_clusters.append(cluster)
        else:
            num_entities_in_old_cluster = len(cluster)
            current_clusters = split_cluster_greedy(
                cluster, signature, pair_distance, entity2svals,
                fake_entity_manager, min_size, min_signature_size
            )

            assert len(cluster) == num_entities_in_old_cluster, "splited cluster: {} is modified (old: {} - new: {})".format(cluster, num_entities_in_old_cluster, len(cluster))
            cassert.test_invalid_min_size_clusters(current_clusters, min_size, "split")
            cassert.test_invalid_signature_size_clusters(current_clusters, entity2svals, fake_entity_manager, min_signature_size, "split")

            logger.debug("splited clusters: {}".format(current_clusters))

            if current_clusters is None or len(current_clusters) == 1:
                logger.debug("Cannot split cluster: {}".format(cluster))
                new_clusters.append(cluster)

                unsplitable_clusters.append(cluster)
                # raise Exception()
            else:
                logger.debug(
                    "split to {} clusters".format(len(current_clusters))
                )
                open_clusters.extend(current_clusters)

        cassert.test_invalid_min_size_clusters(new_clusters, min_size, "splited: {}".format(cluster))
        cassert.test_invalid_signature_size_clusters(new_clusters, entity2svals, fake_entity_manager, min_signature_size, "splited: {}".format(cluster))

    logger.debug("all splitted clusters: {}".format(new_clusters))
    # if len(unsplitable_clusters) > 0:
    #     raise Exception()
    analyze_unsplitable_clusters(unsplitable_clusters, pair_distance, entity2svals, fake_entity_manager, min_size, min_signature_size)


    return new_clusters

def check_splitable_clusters(cluster, sign_freq, min_size, min_signature_size):
    num_entities = len(cluster)

    if num_entities < min_size * 2:
        return False

    sorted_sign_freq = SortedList(sign_freq.items(), key=lambda item: item[1])
    logger.debug("init sorted_sign_freq: {}".format(sorted_sign_freq))
    num_splitable_clusters = 0
    while(len(sorted_sign_freq) >= min_signature_size and num_entities >= min_size * 2):

        new_sorted_sign_freq = SortedList(key=lambda item: item[1])
        for i in range(min_signature_size):
            logger.debug(i)
            sval_id, sval_freq = sorted_sign_freq[i]

            if sval_freq > 1:
                new_sorted_sign_freq.add((sval_id, sval_freq - 1))

            num_entities = num_entities - 1

        sorted_sign_freq = new_sorted_sign_freq
        num_splitable_clusters += 1

        logger.debug("sorted_sign_freq: {}".format(sorted_sign_freq))
        logger.debug("num_splitable_clusters: {}".format(num_splitable_clusters))

    if num_splitable_clusters > 1:
        return True

    return False
    # raise Exception()




def analyze_unsplitable_clusters(unsplitable_clusters, pair_distance, entity2svals, fake_entity_manager, min_size, min_signature_size):
    if len(unsplitable_clusters) == 0:
        return
    logger.debug("analyzing {} unsplitale clusters: {}".format(len(unsplitable_clusters), unsplitable_clusters))

    # raise Exception()
    for cluster in unsplitable_clusters:
        sign_freq = get_sval2freq(cluster, entity2svals, fake_entity_manager)

        logger.debug("unsplitable cluster: {}".format(cluster))
        logger.debug("sign_freq: {}".format(sign_freq))

        is_splitable_cluster = check_splitable_clusters(cluster, sign_freq, min_size, min_signature_size)
        if is_splitable_cluster:
            # raise Exception()
            signature = get_signature(cluster, entity2svals, fake_entity_manager)

            logger.debug("respliting again to analyze")
            current_clusters = split_cluster_greedy(
                cluster, signature, pair_distance, entity2svals,
                fake_entity_manager, min_size, min_signature_size
            )

            logger.debug("sign_freq (len: {}): {}".format(len(sign_freq), sign_freq))
            logger.debug("min_signature_size: {}".format(min_signature_size))
            logger.debug("resulting clusters: {}".format(current_clusters))
            raise Exception()

    # raise Exception()

class ClosestUsers:
    def __init__(self, entity_id, sval_id):
        self.src_entity_id = entity_id
        self.src_sval_id = sval_id

        self.data = {}

    @property
    def sval_ids(self):
        return set(self.data.keys())

    def add_entity_distance(self, entity_id, distance, sval_id):
        sorted_entity_ids = self.data.get(
            sval_id, SortedList(key=lambda item: item[1])
        )

        sorted_entity_ids.add((entity_id, distance))

        if len(sorted_entity_ids) == 1:
            self.data[sval_id] = sorted_entity_ids

    def pop_closest_entities_by_svals(self, sval_ids, num_entities):
        """Select num_entities who are close to self.src_entity_id. Each selection belongs to a unique sval_id which is different from self.src_sval_id.

        Args:
            sval_ids (set[int]): set of sval ids
            num_entities (int): number of selection

        Returns:
            SortedList[(float, int, int)]: sorted list of selections each of which contains a tuple of distance, entity_id and sval_id.
        """
        logger.debug("selecting {} entities among {} sval_ids: {}".format(num_entities, len(sval_ids), sval_ids))
        logger.debug("src_sval_id: {}".format(self.src_sval_id))
        logger.debug("initial data: {}".format(self))

        # closest_entities_in_single_freq_sval = SortedList(key=lambda item: item[0])
        # closest_entities_in_multi_freq_sval = SortedList(key=lambda item: item[0])
        closest_entities = SortedList(key= lambda item: (item[0], item[1]))

        for sval_id in sval_ids:
            entities_by_sval = self.data[sval_id]
            entity_id, dist = entities_by_sval[0]

            if len(entities_by_sval) == 1:
                # closest_entities_in_single_freq_sval.add(item)
                closest_entities.add((1, dist, entity_id, sval_id))
            else:
                # closest_entities_in_multi_freq_sval.add(item)
                closest_entities.add((0, dist, entity_id, sval_id))

            # closest_entities.add((dist, entity_id, sval_id))

        logger.debug(
            "closest_entities (len: {}): {}".format(len(closest_entities), closest_entities)
        )
        # logger.debug(
        #     "closest_entities_in_single_freq_sval (len: {}): {}".format(len(closest_entities_in_single_freq_sval), closest_entities_in_single_freq_sval)
        # )
        # logger.debug(
        #     "closest_entities_in_multi_freq_sval (len: {}): {}".format(len(closest_entities_in_multi_freq_sval), closest_entities_in_multi_freq_sval)
        # )

        selected_entities = SortedList(key=lambda item: item[0])
        # only keeps num_entities selections
        # put priority on selecting entities of sval that have more than 1 entities
        current_index = 0
        while len(selected_entities) < num_entities:
            _, dist, entity_id, sval_id = closest_entities[current_index]
            selected_entities.add((dist, entity_id, sval_id))
            current_index += 1

            logger.debug(
                "selected_entities (len: {}): {}".format(len(selected_entities), selected_entities)
            )
            # pass

        logger.debug(
            "selected_entities (len: {}): {}".format(len(selected_entities), selected_entities)
        )

        # while (len(closest_entities) > num_entities):
        #     dist, entity_id, sval_id = closest_entities.pop(-1)

        # logger.debug("closest entities: {}".format(closest_entities))

        # remove selected entities from self.data
        for _, _, sval_id in selected_entities:
            entities_by_sval = self.data[sval_id]
            entities_by_sval.pop(0)

            if len(entities_by_sval) == 0:
                del self.data[sval_id]

        logger.debug("remaining data: {}".format(self))
        logger.debug("selected_entities: {}".format(selected_entities))

        return selected_entities

    def pop_closest_entities(self, num_entities):
        closest_entities = SortedList(key=lambda item: item[0])

        sorted_svals = SortedList(key=lambda item: (item[0], item[1]))
        for sval_id, entities in self.data.items():
            sorted_svals.add((-len(entities), entities[0][1], sval_id))

        logger.debug("initial data: {}".format(self))
        logger.debug("sorted_svals: {}".format(sorted_svals))
        logger.debug("num_entities: {}".format(num_entities))

        while (len(closest_entities) < num_entities):
            negative_num_entities_in_sval, closest_dist, closest_sval_id = sorted_svals.pop(0)

            entities_by_sval = self.data[closest_sval_id]
            closest_entity_id, closest_dist2 = entities_by_sval.pop(0)

            if len(entities_by_sval) == 0:
                del self.data[closest_sval_id]
            else:
                sorted_svals.add((negative_num_entities_in_sval, entities_by_sval[0][1], closest_sval_id))

            logger.debug(
                "closest entity: {} - dist: {}".format(
                    closest_entity_id, closest_dist2
                )
            )
            logger.debug("sorted_svals: {}".format(sorted_svals))
            logger.debug("current data: {}".format(self))

            closest_entities.add((closest_dist, closest_entity_id))

        return closest_entities

    def __str__(self):
        data_str = ""

        for sval_id, entity_ids in self.data.items():
            entities_str = ",".join(map(lambda item: str(item), entity_ids))
            data_str += "{}:[{}]\n".format(sval_id, entities_str)

        return "num of entities: {} - num of svals: {}\n{}".format(
            len(self), len(self.data), data_str
        )

    def __len__(self):
        count = 0

        for entity_ids in self.data.values():
            count += len(entity_ids)

        return count


def select_entity_randomly(entity_ids):
    entity_id = random.sample(entity_ids, 1)[0]

    return entity_id


def get_first_sval_id(entity_id, entity2svals, fake_entity_manager=None):
    sval_ids = entity2svals.get(entity_id)

    if sval_ids is None:
        if fake_entity_manager is None:
            raise Exception("Cannot find sval of entity_id: {}".format(entity_id))

        sval_id = fake_entity_manager.get_sensitive_value_id(entity_id)
    else:
        sval_id = list(entity2svals[entity_id])[0]
    return sval_id


def generate_closest_entities(
    entity_id, entity_ids, pair_distance, entity2svals, fake_entity_manager
):
    closest_entities = ClosestUsers(
        entity_id, get_first_sval_id(entity_id, entity2svals, fake_entity_manager)
    )

    for entity2_id in entity_ids:
        dist = pair_distance.get_distance(entity_id, entity2_id)
        sval2_id = get_first_sval_id(entity2_id, entity2svals)

        closest_entities.add_entity_distance(entity2_id, dist, sval2_id)

    return closest_entities


def select_farthest_entity(entity_ids, pair_distance):
    entity1_id, entity2_id = select_farthest_entities(entity_ids, pair_distance)

    return random.choice([entity1_id, entity2_id])

def select_farthest_entities(entity_ids, pair_distance):
    max_dist = (None, None, -sys.maxsize)
    for entity1_id, entity2_id in itertools.permutations(entity_ids, r=2):
        dist = pair_distance.get_distance(entity1_id, entity2_id)
        if dist > max_dist[2]:
            max_dist = (entity1_id, entity2_id, dist)

    return max_dist[0], max_dist[1]


def get_signature(cluster, entity2svals, fake_entity_manager):
    signature = set()

    for entity_id in cluster:
        sval_id = get_first_sval_id(entity_id, entity2svals, fake_entity_manager)
        signature.add(sval_id)

    return signature


def get_sval2freq(cluster, entity2svals, fake_entity_manager):
    sval2freq = {}

    for entity_id in cluster:
        sval_id = get_first_sval_id(entity_id, entity2svals, fake_entity_manager)
        sval2freq[sval_id] = sval2freq.get(sval_id, 0) + 1

    return sval2freq

def get_sval2entities(entity_ids, entity2svals, fake_entity_manager):
    sval2entities = {}

    for entity_id in entity_ids:
        sval_id = get_first_sval_id(entity_id, entity2svals, fake_entity_manager)

        entity_ids_set = sval2entities.get(sval_id, None)

        if entity_ids_set is None:
            entity_ids_set = set()
            sval2entities[sval_id] = entity_ids_set

        entity_ids_set.add(entity_id)

    return sval2entities


def split_cluster_greedy(
    cluster, signature, pair_distance, entity2svals, fake_entity_manager,
    min_size, min_signature_size
):
    remaining_entity_ids = cluster.entity_ids.copy()
    logger.debug("initial remaining users: {}".format(remaining_entity_ids))
    # raise Exception()

    remaining_signature = signature.copy()
    logger.debug("initial remaining_signature: {}".format(remaining_signature))

    logger.debug(
        "initial signature freq: {}".format(
            get_sval2freq(cluster, entity2svals, fake_entity_manager)
        )
    )

    new_clusters = []

    while (
        len(remaining_entity_ids) >= min_size and
        len(remaining_signature) >= min_signature_size
    ):
        # randomly select a user
        # current_entity_idx = np.random.randint(0, len(current_entity_ids))
        # current_entity_id = current_entity_ids[current_entity_idx]
        # del current_entity_ids[current_entity_idx]
        current_entity_id = select_farthest_entity(
            remaining_entity_ids, pair_distance
        )
        remaining_entity_ids.remove(current_entity_id)

        logger.debug("current entity: {}".format(current_entity_id))
        logger.debug("remaining entities: {}".format(remaining_entity_ids))

        # init closest users structure
        closest_entity_ids_struct = generate_closest_entities(
            current_entity_id, remaining_entity_ids, pair_distance, entity2svals, fake_entity_manager
        )

        logger.debug(
            "initial closest entities structure: {}".
            format(closest_entity_ids_struct)
        )

        current_cluster = Cluster()
        current_cluster.add_entity(current_entity_id)
        current_signature = {get_first_sval_id(current_entity_id, entity2svals, fake_entity_manager)}

        logger.debug("initial cluster: {}".format(current_cluster))
        logger.debug("initial signature: {}".format(current_signature))

        # add closest users whose svals are different until satisfying min_signature_size
        remaining_sval_ids = closest_entity_ids_struct.sval_ids.difference(
            current_signature
        )
        logger.debug("remaining sval ids: {}".format(remaining_sval_ids))

        # pop entities to ensure that the cluster satisfies signature_min_size
        closest_entity_ids_by_svals = closest_entity_ids_struct.pop_closest_entities_by_svals(
            remaining_sval_ids, min_signature_size - 1
        )
        logger.debug(closest_entity_ids_by_svals)

        for _, entity_id, sval_id in closest_entity_ids_by_svals:
            current_cluster.add_entity(entity_id)
            current_signature.add(sval_id)

        logger.debug("current cluster: {}".format(current_cluster))
        logger.debug(
            "current signature: {}".format(
                get_signature(current_cluster, entity2svals, fake_entity_manager)
            )
        )

        # add closest users to satisfy min_size
        closest_entity_ids = closest_entity_ids_struct.pop_closest_entities(
            max(0, min_size - len(current_cluster))
        )
        logger.debug(
            "found {} entities: {}".format(
                len(closest_entity_ids), closest_entity_ids
            )
        )

        for dist, entity_id in closest_entity_ids:
            current_cluster.add_entity(entity_id)

        new_clusters.append(current_cluster)

        remaining_entity_ids.difference_update(current_cluster)
        remaining_signature = closest_entity_ids_struct.sval_ids

        logger.debug(
            "after: remaining entities: {}".format(remaining_entity_ids)
        )
        logger.debug(
            "after: remaining signature: {}".format(remaining_signature)
        )
        logger.debug(
            "after: remaining signature freq: {}".format(
                get_sval2freq(remaining_entity_ids, entity2svals, fake_entity_manager)
            )
        )

        logger.debug("new cluster: {}".format(current_cluster))
        logger.debug(
            "new cluster signature freq: {}".format(
                get_sval2freq(current_cluster, entity2svals, fake_entity_manager)
            )
        )

    logger.debug("all new clusters: {}".format(new_clusters))
    for entity_id in remaining_entity_ids:
        best_cluster_dist = (sys.maxsize, Cluster())

        for new_cluster in new_clusters:
            cluster_dist = -sys.maxsize
            for entity2_id in cluster:
                dist = pair_distance.get_distance(entity_id, entity2_id)

                cluster_dist = max(cluster_dist, dist)

            if cluster_dist < best_cluster_dist[0]:
                best_cluster_dist = (cluster_dist, new_cluster)

        best_cluster_dist[1].add_entity(entity_id)

    logger.debug("final clusters: {}".format(new_clusters))
    # if len(new_clusters) == 1 and len(cluster) >= min_size * 2:
    #     raise Exception()

    # raise Exception()
    return new_clusters