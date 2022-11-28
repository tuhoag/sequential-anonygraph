from anonygraph.algorithms.pair_distance import PairDistance
from typing import DefaultDict
from anonygraph.algorithms.clustering.enforcers.base_enforcer import BaseEnforcer
import itertools
from anonygraph.utils.runner import string2range
from sortedcontainers import SortedList
from networkx.algorithms.structuralholes import effective_size
import numpy as np
import random
import sys
from anonygraph.algorithms import Cluster, fake_entity_manager
import math
import logging

from anonygraph.info_loss import info
from anonygraph.assertions import clusters_assertions as cassert
from .base_enforcer import BaseEnforcer


logger = logging.getLogger(__name__)

def remove_invalid_clusters(clusters, entity2svals, min_size, min_num_sensitive_vals):
    new_clusters = []
    entity_ids = []

    for cluster in clusters:
        if len(cluster) < min_size or len(info.get_generalized_signature_info_from_dict(entity2svals, cluster)) < min_num_sensitive_vals:
            entity_ids.extend(cluster)
        else:
            new_clusters.append(cluster)

    return new_clusters, entity_ids

def calculate_distance_from_entity_to_cluster(entity_id, cluster, pair_distance):
    distance = -sys.maxsize

    # logger.debug('{} - {} - initial distance: {}'.format(entity_id, cluster, distance))

    for cluster_entity_id in cluster:
        entity_to_entity_distance = pair_distance.get_distance(entity_id, cluster_entity_id)

        distance = max(entity_to_entity_distance, distance)

        # logger.debug('{} - {} -  distance: {} - {}'.format(entity_id, cluster_entity_id, entity_to_entity_distance, distance))

    return distance

def assign_valid_clusters(clusters, entity_ids, pair_distance, entity2svals, fake_entity_manager, max_dist, min_size):
    """Add an entity to a cluster if distance <= max_dist and its sensitive value is in cluster signature.

    Args:
        clusters (list of clusters): List of valid signature clusters
        entity_ids (list of entities' ids): List of removed entity ids
        pair_distance (PairDistance): pairwise distance of all entities
        entity2svals (dict): Mapping between entity id and its sensitive value
        max_dist (float): Maximum distance between an entity and a cluster
    """
    count = len(entity_ids)

    logger.debug("assigning {} clusters and {} entities".format(len(clusters), len(entity_ids)))
    index = 0
    while index < len(entity_ids):
        entity_id = entity_ids[index]
        closest_cluster = (max_dist, None)
        svals = entity2svals[entity_id]

        # find clusters that can contain current entity
        for cluster in clusters:
            signature = info.get_generalized_signature_info_from_dict(entity2svals, fake_entity_manager, cluster)

            # logger.debug("signature: {} - svals: {} (contain: {})".format(signature, svals, svals.issubset(signature)))
            # # raise Exception()
            if svals.issubset(signature):
                distance = calculate_distance_from_entity_to_cluster(entity_id, cluster, pair_distance)

                # logger.debug('entity: {} cluster: {} - dist: {} - max_dist: {} - closest: {}'.format(entity_id, cluster, distance, max_dist, closest_cluster))
                if distance <= closest_cluster[0]:
                    closest_cluster = (distance, cluster)

        if closest_cluster[1] is not None:
            # logger.debug("add {} to {}".format(entity_id, closest_cluster[1]))
            closest_cluster[1].add_entity(entity_id)
            entity_ids.pop(index)
            # count += 1
        else:
            index += 1

    cassert.test_invalid_min_size_clusters(clusters, min_size)

    logger.debug("merged {} entities.".format(count - len(entity_ids)))


def calculate_cluster_signature_stats(cluster, entity2svals, fake_entity_manager):
    signature_list = list(info.get_generalized_signature_info_from_dict(entity2svals, fake_entity_manager, cluster))
    freq_list = np.zeros(len(signature_list))
    val2entity_list = [set() for _ in range(len(signature_list))]

    for entity_id in cluster:
        sval = list(entity2svals[entity_id])[0]
        sindex = signature_list.index(sval)

        freq_list[sindex] += 1
        val2entity_list[sindex].add(entity_id)

    return signature_list, freq_list, val2entity_list

def count_splittable_clusters(cluster, freq_list, min_size):
    smallest_sindex = np.argmin(freq_list)
    num_signature_splitable_clusters = int(freq_list[smallest_sindex])

    num_size_splitable_clusters = math.floor(len(cluster) / min_size)

    num_splitable_clusters = min(num_signature_splitable_clusters, num_size_splitable_clusters)

    logger.debug("num splitable: {} <- num splitable size: {} (min_size: {}) - num spitable signature: {}".format(num_splitable_clusters, num_size_splitable_clusters, min_size, num_signature_splitable_clusters))

    return num_splitable_clusters

def split_same_signature_cluster(cluster, num_clusters, pair_distance, signature_list, val2entity_list, freq_list):
    smallest_sindex = np.argmin(freq_list)
    remaining_entity_ids = cluster.entity_ids
    smallest_svalues_entity_ids = list(val2entity_list[smallest_sindex])

    if num_clusters > len(smallest_svalues_entity_ids):
        raise Exception("Cannot split {} to {} clusters without making its size invalid".format(cluster, num_clusters))

    new_clusters = []

    logger.debug("initial entities in cluster (len: {}): {}".format(len(remaining_entity_ids), remaining_entity_ids))
    selected_entity_idxes = random.sample(range(len(smallest_svalues_entity_ids)), num_clusters)
    selected_entity_ids = list(map(lambda idx: smallest_svalues_entity_ids[idx], selected_entity_idxes))
    logger.debug("selected entity ids: {}".format(selected_entity_ids))

    for entity_id in selected_entity_ids:
        new_clusters.append(Cluster.from_iter([entity_id]))
        remaining_entity_ids.remove(entity_id)

    logger.debug("val2entity list: {}".format(val2entity_list))
    logger.debug("initial new clusters: {}".format(new_clusters))
    logger.debug("remaining entity ids after initializing clusters (len: {}): {}".format(len(remaining_entity_ids), remaining_entity_ids))

    for new_cluster in new_clusters:
        for sid, entity_ids in enumerate(val2entity_list):
            if sid != smallest_sindex:
                logger.debug("before entity ids (len: {}): {}".format(len(entity_ids), entity_ids))
                closest_entity = (sys.maxsize, None)
                for entity_id in entity_ids:

                    dist = calculate_distance_from_entity_to_cluster(entity_id, new_cluster, pair_distance)

                    if dist < closest_entity[0]:
                        closest_entity = (dist, entity_id)

                if closest_entity[1] is not None:
                    logger.debug("adding closest entity: {} to cluster: {}".format(closest_entity, new_cluster))

                    new_cluster.add_entity(closest_entity[1])
                    entity_ids.remove(closest_entity[1])
                    remaining_entity_ids.remove(closest_entity[1])

                logger.debug("after entity ids (len: {}): {}".format(len(entity_ids), entity_ids))

    logger.debug("remaining entity ids after splitting (len: {}): {}".format(len(remaining_entity_ids), remaining_entity_ids))
    logger.debug("Valid signature clusters: {}".format(new_clusters))

    current_new_clusters = list(new_clusters)

    for entity_id in remaining_entity_ids:
        closest_cluster = (sys.maxsize, None)
        for new_cluster_idx, new_cluster in enumerate(current_new_clusters):
        # for current_cluster_idx in range(new_cluster_idx, len(new_clusters)):
            # new_cluster = new_clusters[current_cluster_idx]
            dist = calculate_distance_from_entity_to_cluster(entity_id, new_cluster, pair_distance)

            if dist < closest_cluster[0]:
                closest_cluster = (dist, new_cluster_idx)

        if closest_cluster[1] is not None:
            logger.debug("adding entity id: {} to cluster: {} - dist: {}".format(entity_id, current_new_clusters[closest_cluster[1]], closest_cluster[0]))

            current_new_clusters[closest_cluster[1]].add_entity(entity_id)
            current_new_clusters.pop(closest_cluster[1])

            if len(current_new_clusters) == 0:
                current_new_clusters = list(new_clusters)


    # logger.debug("final remaining entity ids (len: {}): {}".format(len(remaining_entity_ids), remaining_entity_ids))
    logger.debug("Final clusters: {}".format(new_clusters))

    return new_clusters


def split_same_signature_clusters(clusters, entity2svals, fake_entity_manager, pair_distance, min_size, max_size, min_signature_size):
    """Split clusters such that the resulting clusters are still valid after splitting. If they cannot be splitted, add them to the result.
    """
    new_clusters = []

    for cluster in clusters:
        # if cluster can be split with same signature
        # signature_list, freq_list, val2entity_list = calculate_cluster_signature_stats(cluster, entity2svals, fake_entity_manager)
        # num_splitable_clusters = count_splittable_clusters(cluster, freq_list, min_size)
        # if num_splitable_clusters > 1:
        signature = info.get_generalized_signature_info_from_dict(entity2svals, fake_entity_manager, cluster)
        current_clusters = split_cluster_to_same_signature_clusters(cluster, signature, pair_distance, entity2svals, fake_entity_manager, min_size, min_signature_size)
        # current_clusters = split_same_signature_cluster(cluster, num_splitable_clusters, pair_distance, signature_list, val2entity_list, freq_list)

        new_clusters.extend(current_clusters)
        # else:
        #     new_clusters.append(cluster)

    return new_clusters

def find_signatures(signature_list, entity_ids, pair_distance, entity2svals, min_num_svals):
    signature_assignment = []
    num_signatures = math.floor(len(signature_list) / min_num_svals)

    while len(signature_assignment) < len(signature_list):
        cindex = min(math.floor(len(signature_assignment) / min_num_svals), num_signatures - 1)

        signature_assignment.append(cindex)


    logger.debug("l: {} - assignment: {}".format(min_num_svals, signature_assignment))
    return signature_assignment

def create_clusters_based_on_signature_assignment(signature_assignment, entity_ids, entity2svals, signature_list):
    clusters = [Cluster() for _ in range(max(signature_assignment) + 1)]

    logger.debug("signature list: {}".format(signature_list))
    logger.debug("signature assignment: {}".format(signature_assignment))
    logger.debug("num clusters: {}".format(len(clusters)))
    for entity_id in entity_ids:
        svals = entity2svals[entity_id]
        sval = list(svals)[0]

        sindex = signature_list.index(sval)
        logger.debug("sval '{}' with index {} in {}".format(sval, sindex, signature_list))

        cid = signature_assignment[sindex]
        logger.debug("cluster id: {}".format(cid))
        clusters[cid].add_entity(entity_id)

    return clusters

def split_cluster_to_same_signature_clusters(cluster, signature, pair_distance, entity2svals, fake_entity_manager, min_size, min_signature_size):
    # count splitable clusters
    signature_list, freq_list, val2entity_list = calculate_cluster_signature_stats(cluster, entity2svals, fake_entity_manager)
    num_splitable_clusters = count_splittable_clusters(cluster, freq_list, min_size)

    new_clusters = []

    if num_splitable_clusters > 1:
        logger.debug("spliting cluster {} to {} clusters with the same signature".format(cluster, num_splitable_clusters))
        current_clusters = split_same_signature_cluster(cluster, num_splitable_clusters, pair_distance, signature_list, val2entity_list, freq_list)

        new_clusters.extend(current_clusters)
    else:
        new_clusters.append(cluster)

    return new_clusters

def init_signature_assignment(signature_list, num_signatures, min_signature_size):
    signature_assignments = []
    initial_assignment = []

    for cid in range(num_signatures):
        for _ in range(min_signature_size):
            initial_assignment.append(cid)

    logger.debug("initial assignment: {}".format(initial_assignment))

    num_remaining = len(signature_list) - len(initial_assignment)
    logger.debug("num remaining: {}".format(num_remaining))

    for arr_remaining in itertools.product(range(num_signatures), repeat=num_remaining):
        current_assignment = initial_assignment + list(arr_remaining)
        logger.debug("current: {} - remaining: {}".format(current_assignment, arr_remaining))

        signature_assignments.append(current_assignment)

    logger.debug(signature_assignments)

    # for sid in range(len(signature_assignment)):
    #     cid = min(math.floor(sid / min_signature_size), num_signatures - 1)
    #     logger.debug("cid: {} - sid / {}: {}".format(cid, min_signature_size, math.floor(sid / num_signatures)))
    #     signature_assignment[sid] = cid

    # logger.debug("signature list: {} - num clusters: {} -> init assignment: {}".format(signature_list, num_signatures, signature_assignment))
    # if len(signature_assignments) > 1:
    #     raise Exception()

    return signature_assignments


def get_clusters_freq_stats(cluster, entity2svals):
    s2entities = {}
    s2num_entities = {}

    for entity_id in cluster:
        sval_id = list(entity2svals[entity_id])[0]
        entity_ids = s2entities.get(sval_id, [])
        s2entities[sval_id] = entity_ids
        entity_ids.append(entity_id)

        count = s2num_entities.get(sval_id, 0)
        s2num_entities[sval_id] = count + 1

    return s2entities, s2num_entities

def calculate_signature_assignment_cost(signature_assignment, signature_list, num_clusters, s2entities, s2num_entities, pair_distance):
    c2entities = {}
    for sid, cid in enumerate(signature_assignment):
        entity_ids = c2entities.get(cid, [])

        sval_id = signature_list[sid]
        sentity_ids = s2entities[sval_id]

        entity_ids.extend(sentity_ids)
        c2entities[cid] = entity_ids

    logger.debug("assignment: {} - c2entities: {}".format(signature_assignment, c2entities))

    sum_dists = 0

    for cid in range(num_clusters):
        entity_ids = c2entities[cid]

        logger.debug("cluster: {}".format(entity_ids))

        max_dist = 0
        for entity1_idx in range(len(entity_ids)):
            for entity2_idx in range(entity1_idx, len(entity_ids)):
                entity1_id = entity_ids[entity1_idx]
                entity2_id = entity_ids[entity2_idx]

                dist = pair_distance.get_distance(entity1_id, entity2_id)
                max_dist = max(dist, max_dist)
                # logger.debug("entities: {}, {} - dist: {} - max dist: {}".format(entity1_id, entity2_id, dist, max_dist))

        sum_dists += max_dist * len(entity_ids)

    logger.debug("num_dists: {}".format(sum_dists))
    return sum_dists

def check_valid_min_size_signature_assignment(signature_assignment, signature_list, num_clusters, s2entities, s2num_entities, pair_distance, min_size):
    c2num_entities = {}
    for sid, cid in enumerate(signature_assignment):
        count = c2num_entities.get(cid, 0)

        sval_id = signature_list[sid]
        count += s2num_entities[sval_id]

        c2num_entities[cid] = count


    current_min_size = min(c2num_entities.values())
    logger.debug("current min size: {} - min size: {} - feq: {} - s2num: {} - assignment: {}".format(current_min_size, min_size, c2num_entities, s2num_entities, signature_assignment))
    return current_min_size >= min_size

def get_entity2sval_matrix(entity_ids, signature_list, entity2svals):
    entity2sval_matrix = np.zeros((len(entity_ids), len(signature_list)), dtype=int)

    for entity_idx, entity_id in enumerate(entity_ids):
        sval_ids = entity2svals[entity_id]

        for sval_id in sval_ids:
            sval_idx = signature_list.index(sval_id)

            entity2sval_matrix[entity_idx,sval_idx] = 1

    return entity2sval_matrix


def calculate_e2e_matrix_cost(e2e_matrix, dist_matrix):
    e2e_dist_matrix = np.multiply(e2e_matrix, dist_matrix)
    max_dist_cluster_arr = np.max(e2e_dist_matrix, axis=0)
    num_entities_cluster_arr = np.sum(e2e_matrix, axis=0)

    cost_cluster_arr = np.multiply(max_dist_cluster_arr, num_entities_cluster_arr)
    result = np.sum(cost_cluster_arr)

    logger.debug("e2e matrix: {}".format(e2e_matrix))
    logger.debug("dist matrix: {}".format(dist_matrix))
    logger.debug("e2e_dist: {}".format(e2e_dist_matrix))
    logger.debug("max dist: {}".format(max_dist_cluster_arr))
    logger.debug("num e cluster: {}".format(num_entities_cluster_arr))
    logger.debug("cost cluster: {}".format(cost_cluster_arr))
    logger.debug("cost: {}".format(result))

    # raise Exception()

    return result


def check_valid_e2e_matrix(e2e_matrix, e2s_matrix, min_size, min_signature_size):
    num_entities_cluster_arr = np.sum(e2e_matrix, axis=0)
    min_cluster_size = np.min(num_entities_cluster_arr)

    e2sfreq_matrix = np.dot(e2e_matrix, e2s_matrix)
    e2sig_matrix = e2sfreq_matrix >= 1
    e2sigsize_arr = np.sum(e2sig_matrix, axis=1)
    min_cluster_sig_size = np.min(e2sigsize_arr)

    logger.debug("num entities cluster: {}".format(num_entities_cluster_arr))
    logger.debug("min cluster size: {}".format(min_cluster_size))
    logger.debug("e2sigfreq: {}".format(e2sfreq_matrix))
    logger.debug("e2sig: {}".format(e2sig_matrix))
    logger.debug("min cluster sig size: {}".format(min_cluster_sig_size))

    result = min_cluster_size >= min_size and min_cluster_sig_size >= min_signature_size

    logger.debug("valid: {}".format(result))
    # raise Exception()
    return result



def hash_e2e_matrix(e2e_matrix):
    result = e2e_matrix.flatten()
    logger.debug("matrix shape: {} -> result shape: {}".format(e2e_matrix.shape, result.shape))
    result = tuple(result)
    # logger.debug(result)
    return result


def find_optimal_subclusters_by_minimizing_cost(cluster, signature_list, pair_distance, entity2svals, min_size, min_signature_size):
    entity_ids = list(cluster)
    num_entities = len(entity_ids)
    logger.debug("entity ids ({}): {}".format(num_entities, entity_ids))

    initial_entity2entity_matrix = np.zeros((num_entities,num_entities), dtype=int)
    logger.debug("initial entity2entity matrix: {}".format(initial_entity2entity_matrix.shape))

    entity2sval_matrix = get_entity2sval_matrix(entity_ids, signature_list, entity2svals)
    logger.debug("entity2sval matrix: {}".format(entity2sval_matrix.shape))

    dist_matrix = pair_distance.get_distance_matrix(entity_ids)
    logger.debug("distance matrix {}: {}".format(dist_matrix.shape, dist_matrix))

    initial_hash = hash_e2e_matrix(initial_entity2entity_matrix)
    closed_state_hashes = { initial_hash }
    opened_state_hashes = { initial_hash }
    open_states = [ initial_entity2entity_matrix ]

    best_state = (sys.maxsize, None)
    count = 0

    total_num_states = 2**((num_entities * num_entities - num_entities)/2)

    # raise Exception("{} - {}".format(num_entities, total_num_states))
    while(len(open_states) > 0):
        count += 1
        logger.info("num open states: {}/{} ({}) - closed states: {} - best state: {}".format(len(open_states), total_num_states, len(open_states)/total_num_states*100, len(closed_state_hashes), best_state))
        current_state = open_states.pop(0)

        cost = calculate_e2e_matrix_cost(current_state, dist_matrix)

        # if count == 3:
        #     raise Exception()
        if check_valid_e2e_matrix(current_state, entity2sval_matrix, min_size, min_signature_size):
            logger.debug("valid")
            if cost < best_state[0]:
                best_state = (cost, current_state)

        for entity1_idx, entity2_idx in itertools.combinations(range(num_entities), r=2):
            # logger.debug("{}, {}".format(entity1_idx, entity2_idx))
            next_state = current_state.copy()
            next_state[entity1_idx, entity2_idx] = (current_state[entity1_idx, entity2_idx] + 1) % 2
            next_state[entity2_idx, entity1_idx] = (current_state[entity1_idx, entity2_idx] + 1) % 2

            next_state_hash = hash_e2e_matrix(next_state)

            if next_state_hash not in closed_state_hashes and next_state_hash not in opened_state_hashes:
                closed_state_hashes.add(next_state_hash)
                opened_state_hashes.add(next_state_hash)

                open_states.append(next_state)

        # logger.info(len(open_states) >= 3)
        # if len(open_states) >= 200:
        #     logger.info("opened: {} - closed: {}".format(len(open_states), len(closed_state_hashes)))
        #     raise Exception()

        # raise Exception()

    logger.debug("found best solution: {}".format(best_state))
    raise Exception()



def find_signatures_by_minimizing_cost(cluster, signature_list, pair_distance, entity2svals, min_size, min_signature_size):
    # find signatures such that all users are in the same
    num_signatures = math.floor(len(signature_list)/ min_signature_size)
    num_clusters_by_size = math.floor(len(cluster) / min_size)
    num_clusters = min(num_clusters_by_size, num_signatures)

    # while (num_clusters >= 2):

    s2entities, s2num_entities = get_clusters_freq_stats(cluster, entity2svals)

    # current_cost = calculate_signature_assignment_cost(signature_assignment, signature_list, num_clusters, s2entities, s2num_entities, pair_distance)

    # logger.debug("initial assignment: ({}, {})".format(current_cost, signature_assignment))
    # raise Exception()
    # if check_valid_min_size_signature_assignment(signature_assignment, signature_list, num_clusters, s2entities, s2num_entities, pair_distance, min_size):
    #     current_best = (current_cost, signature_assignment)
    # else:



    while num_clusters > 1:
        logger.debug("splitting to {} clusters...".format(num_clusters))
        logger.debug("num clusters: {} <- by size: {} - by signature: {}".format(num_clusters, num_clusters_by_size, num_signatures))

        signature_assignments = init_signature_assignment(signature_list, num_clusters, min_signature_size)

        current_best = (sys.maxsize, None)
        count = 0

        closed_assignments = set()
        for signature_assignment in signature_assignments:
            for current_assignment in itertools.permutations(signature_assignment):
                # current assignment is a tuple
                logger.debug(current_assignment)
                # if len(signature_assignments) > 1:
                #     raise Exception()

                if current_assignment not in closed_assignments:
                    if check_valid_min_size_signature_assignment(current_assignment, signature_list, num_clusters, s2entities, s2num_entities, pair_distance, min_size):
                        current_cost = calculate_signature_assignment_cost(current_assignment, signature_list, num_clusters, s2entities, s2num_entities, pair_distance)

                        logger.debug("valid assignment: ({}, {}) - current best: {}".format(current_cost, current_assignment, current_best))

                        # update if cost is lower
                        if current_cost < current_best[0]:
                            logger.debug("update best: {} -> {}".format(current_best, (current_cost, current_assignment)))
                            current_best = (current_cost, current_assignment)

                    else:
                        logger.debug("invalid assignment: {}".format(current_assignment))

                count += 1
                closed_assignments.add(current_assignment)

                logger.debug("current assignment itertools ({}): {} - closed assignments: {}".format(count, current_assignment,closed_assignments))
                # else:
                #     logger.debug("existed assignment: {} - closed assignments: {}".format(current_assignment, closed_assignments))

        if current_best[1] is None:
            num_clusters = num_clusters - 1
        else:
            break

    logger.debug("found the best: {}".format(current_best))
    # if num_clusters >= 3:
    # raise Exception()
    return current_best[1]


def split_cluster_to_different_signature_clusters(cluster, signature, pair_distance, entity2svals, fake_entity_manager, min_size, min_signature_size):
    # find signatures such that all users are in the same
    signature_list = list(signature)
    signature_assignment = find_signatures_by_minimizing_cost(cluster, signature_list, pair_distance, entity2svals, min_size, min_signature_size)

    if signature_assignment is not None:
        new_clusters = create_clusters_based_on_signature_assignment(signature_assignment, cluster, entity2svals, signature_list)
    else:
        new_clusters = [cluster]

    logger.debug("new clusters: {}".format(new_clusters))

    # if len(new_clusters) == 1:
    #     raise Exception()
    # raise Exception()
    return new_clusters

def build_minimum_valid_clusters(clusters, pair_distance, entity2svals, fake_entity_manager, min_size, min_signature_size):
    new_clusters = []
    removed_entity_ids = []

    for cluster in clusters:
        signature = info.get_generalized_signature_info_from_dict(entity2svals, fake_entity_manager, cluster)

        cluster_size = len(cluster)
        signature_size = len(signature)

        logger.debug("cluster size: {} - signature_size: {} - min_size: {} - min_signature_size: {}".format(cluster_size, signature_size, min_size, min_signature_size))
        status_str = ""
        if signature_size < min_signature_size:
            status_str += "invalid signature size: {} (min_signature_size: {}) - ".format(signature_size, min_signature_size)
        elif min_signature_size <= signature_size < min_signature_size * 2:
            status_str += "valid signature size: {} (min_signature_size: {}) - ".format(signature_size, min_signature_size)
        else:
            status_str += "too big signature size: {} (min_signature_size: {}) - ".format(signature_size, min_signature_size)

        if cluster_size < min_size:
            status_str += "invalid cluster size: {} (min_size: {})".format(cluster_size, min_size)
        elif min_size <= cluster_size < min_size * 2:
            status_str += "valid cluster size: {} (min_size: {})".format(cluster_size, min_size)
        else:
            status_str += "too big cluster size: {} (min_size: {})".format(cluster_size, min_size)

        logger.debug(status_str)

        if signature_size < min_signature_size:
            if cluster_size < min_size:
                logger.debug("remove cluster: {}".format(cluster))
                removed_entity_ids.extend(cluster)
            elif min_size <= cluster_size < min_size * 2:
                logger.debug("remove cluster: {}".format(cluster))
                removed_entity_ids.extend(cluster)
            else:
                # split this cluster will create invalid signature size clusters
                # logger.debug("split based on cluster size and add fake entities whose s values are missed")
                logger.debug("remove cluster: {}".format(cluster))
                removed_entity_ids.extend(cluster)
                # current_clusters = split_with_same_size_kmedoids(cluster, min_size)
                # add_fake_entities_with_random_svals(current_clusters, entity2svals, fake_entity_manager, min_signature_size)

                # new_clusters.extend(current_clusters)
                # raise Exception("split based on cluster size")

        elif min_signature_size <= signature_size < min_signature_size * 2:
            if cluster_size < min_size:
                logger.debug("remove cluster: {}".format(cluster))
                removed_entity_ids.extend(cluster)
            elif min_size <= cluster_size < min_size * 2:
                logger.debug("add cluster: {}".format(cluster))
                new_clusters.append(cluster)
            else:
                # need to count splitable signatures and split based on the counted value.
                # for example, |c| = 8, |sign(c)|=2, l = 2, k = 3
                logger.debug("split same signature cluster and add fake entities whose s values are missed")
                current_clusters = split_cluster_to_same_signature_clusters(cluster, signature, pair_distance, entity2svals, fake_entity_manager, min_size, min_signature_size)
                new_clusters.extend(current_clusters)
                # raise Exception()
        else:
            if cluster_size < min_size:
                # logger.debug("split signature: {} - signature size: {}".format(signature, signature_size))
                logger.debug("remove cluster: {}".format(cluster))
                removed_entity_ids.extend(cluster)
                # raise Exception()
            elif min_size <= cluster_size < min_size * 2:
                logger.debug("add cluster: {}".format(cluster))
                new_clusters.append(cluster)
            else:
                # find optimal signatures such that minimizing the max_size
                # split same signature
                logger.debug("split cluster: {} with different signatures - signature size: {}".format(cluster, signature))
                current_clusters = split_cluster_to_different_signature_clusters(cluster, signature, pair_distance, entity2svals, fake_entity_manager, min_size, min_signature_size)
                new_clusters.extend(current_clusters)

                # raise Exception()

    return new_clusters, removed_entity_ids


def calculate_real_max_dist(pair_distance: PairDistance, max_dist: float) -> float:
    max_real_dist = pair_distance.max_distance
    min_real_dist = pair_distance.min_distance
    real_threshold = max_dist * (max_real_dist - min_real_dist) + min_real_dist

    logger.debug("max dist: {} - min dist: {} - threshold {} -> {}".format(max_real_dist, min_real_dist, max_dist, real_threshold))
    # raise Exception()
    return real_threshold


def add_fake_entities(clusters, entity2sval, fake_entity_manager, min_size, min_signature_size):
    count = 0
    for cluster in clusters:
        if len(cluster) == 0:
            raise Exception()

        num_required_entities = max(0, min_size - len(cluster))

        if num_required_entities == 0:
            continue

        logger.debug("num required fake entities: {}".format(num_required_entities))

        # find freq of sensitive values
        signature = info.get_generalized_signature_info_from_dict(entity2sval, fake_entity_manager, cluster)
        signature_freq_dict = {}
        for value_id in signature:
            freq = signature_freq_dict.get(value_id, 0)
            freq += 1
            signature_freq_dict[value_id] = freq

        # sorted s by freq
        freq_sorted_list = SortedList(signature_freq_dict.items(), key=lambda item: item[1])

        logger.debug("initial freq: {}".format(freq_sorted_list))

        # count = 0
        for _ in range(num_required_entities):
            # take the smallest
            logger.debug(freq_sorted_list)
            least_frequent_value_id, least_freq = freq_sorted_list.pop(0)
            # logger.debug(least_frequent_value_id)

            # add fake entity with the least frequent sensitive value
            fake_entity_id = fake_entity_manager.create_new_fake_entity(least_frequent_value_id)

            # update freq set
            freq_sorted_list.add((least_frequent_value_id, least_freq + 1))
            count += 1

        logger.debug("final freq: {}".format(freq_sorted_list))
        # logger.debug("added {} fake entities".format(count))
        # raise Exception()




    logger.debug("added {} fake entities".format(count))
    # raise Exception()



def build_valid_signatures_clusters_for_updation(clusters, pair_distance, entity2svals, fake_entity_manager, min_size, min_signature_size, historical_table):
    """Group entities in clusters to new clusters such that entites in the same cluster have the same signature in the previous anonymized graph. Then, remove clusters whose entities are not enough to keep their entities' signature identical to the previous one.
    """
    result = {}
    for cluster in clusters:
        for entity_id in cluster:
            signature_str = str(historical_table.get_signature(entity_id))

            clusters_entry = result.get(signature_str, [])
            clusters_entry.append(entity_id)
            result[signature_str] = clusters_entry

    new_clusters = []
    removed_entity_ids = []
    logger.debug("result: {}".format(result))
    logger.debug("entity2svals: {}".format(entity2svals))
    for pre_signature_str, cluster in result.items():
        logger.debug("cluster: {}".format(cluster))
        # for cluster in clusters:
        signature = info.get_generalized_signature_info_from_dict(entity2svals, fake_entity_manager, cluster)

        if str(signature) != pre_signature_str or len(cluster) < min_size:
            removed_entity_ids.extend(cluster)
            continue

        new_clusters.append(Cluster.from_iter(cluster))

    return new_clusters, removed_entity_ids

def get_cluster_status_str(cluster_size, signature_size, min_size, min_signature_size):
    status_str = ""
    if signature_size < min_signature_size:
        status_str += "invalid signature size: {} (min_signature_size: {}) - ".format(signature_size, min_signature_size)
    elif min_signature_size <= signature_size < min_signature_size * 2:
        status_str += "valid signature size: {} (min_signature_size: {}) - ".format(signature_size, min_signature_size)
    else:
        status_str += "too big signature size: {} (min_signature_size: {}) - ".format(signature_size, min_signature_size)

    if cluster_size < min_size:
        status_str += "invalid cluster size: {} (min_size: {})".format(cluster_size, min_size)
    elif min_size <= cluster_size < min_size * 2:
        status_str += "valid cluster size: {} (min_size: {})".format(cluster_size, min_size)
    else:
        status_str += "too big cluster size: {} (min_size: {})".format(cluster_size, min_size)

    return status_str


def remove_invalid_size_and_signature_clusters(clusters, pair_distance, entity2svals, fake_entity_manager, min_size, min_signature_size):
    new_clusters = []
    removed_entity_ids = []

    for cluster in clusters:
        signature = info.get_generalized_signature_info_from_dict(entity2svals, fake_entity_manager, cluster)

        cluster_size = len(cluster)
        signature_size = len(signature)

        logger.debug("cluster size: {} - signature_size: {} - min_size: {} - min_signature_size: {}".format(cluster_size, signature_size, min_size, min_signature_size))

        logger.debug(get_cluster_status_str(cluster_size, signature_size, min_size, min_signature_size))

        if signature_size >= min_signature_size and cluster_size >= min_size:
            new_clusters.append(cluster)
        else:
            removed_entity_ids.extend(cluster)

    return new_clusters, removed_entity_ids

def split_big_clusters(clusters, pair_distance, entity2svals, fake_entity_manager, min_size, min_signature_size):
    new_clusters = []

    open_clusters = [cluster for cluster in clusters]

    while len(open_clusters) > 0:
        cluster = open_clusters.pop(0)

        signature = info.get_generalized_signature_info_from_dict(entity2svals, fake_entity_manager, cluster)

        cluster_size = len(cluster)
        signature_size = len(signature)

        logger.info("current cluster (num: {}): {}".format(len(open_clusters), cluster))
        logger.debug(get_cluster_status_str(cluster_size, signature_size, min_size, min_signature_size))

        if cluster_size < min_size or signature_size < min_signature_size:
            raise Exception("cluster (size: {} - sig size: {}): {} is invalid (min_size: {} - min_sig_size: {}".format(cluster_size, signature_size, cluster, min_size, min_signature_size))

        current_clusters = [cluster]

        if signature_size < 2 * min_signature_size:
            if cluster_size < 2 * min_size:
                logger.info("add cluster")
                new_clusters.append(cluster)
            else:
                # split same signature clusters

                current_clusters = split_cluster_to_same_signature_clusters(cluster, signature, pair_distance, entity2svals, fake_entity_manager, min_size, min_signature_size)
                logger.info("split same signature to {} clusters".format(len(current_clusters)))
                logger.debug("splited clusters: {}".format(current_clusters))
                # open_clusters.extend(current_clusters)
        else:
            if cluster_size < 2 * min_size:
                logger.info("add cluster")
                new_clusters.append(cluster)
            else:
                # split different signature clusters
                # logger.info("split different signature clusters")
                current_clusters = split_cluster_to_different_signature_clusters(cluster, signature, pair_distance, entity2svals, fake_entity_manager, min_size, min_signature_size)
                logger.info("split different signature to {} clusters".format(len(current_clusters)))
                logger.debug("splited clusters: {} - origin: {}".format(current_clusters, cluster))
                # open_clusters.extend(current_clusters)

        if len(current_clusters) > 1:
            open_clusters.extend(current_clusters)
        else:
            new_clusters.append(cluster)

    return new_clusters

class MergeSplitAssignmentEnforcer(BaseEnforcer):
    def __init__(self, min_size, min_num_sensitive_vals, max_dist):
        self.min_size = min_size
        self.min_signature_size = min_num_sensitive_vals
        self.max_dist = max_dist

    def __call__(self, clusters, pair_distance, entity2svals, fake_entity_manager):
        real_max_dist = calculate_real_max_dist(pair_distance, self.max_dist)
        logger.debug("real max dist: {}".format(real_max_dist))


        new_clusters, removed_entity_ids = remove_invalid_size_and_signature_clusters(clusters, pair_distance, entity2svals, fake_entity_manager, self.min_size, self.min_signature_size)
        cassert.test_invalid_signature_size_clusters(new_clusters, entity2svals, fake_entity_manager, self.min_signature_size)
        cassert.test_invalid_min_size_clusters(new_clusters, self.min_size)



        # new_clusters, removed_entity_ids = build_minimum_valid_clusters(clusters, pair_distance, entity2svals, fake_entity_manager, self.min_size, self.min_signature_size)
        # cassert.test_invalid_signature_size_clusters(new_clusters, entity2svals, fake_entity_manager, self.min_signature_size)
        # cassert.test_invalid_min_size_clusters(new_clusters, self.min_size)

        # logger.debug("there are {} valid signature clusters with {} removed entities".format(len(new_clusters), len(removed_entity_ids)))

        num_removed_entities = len(removed_entity_ids)
        assign_valid_clusters(new_clusters, removed_entity_ids, pair_distance, entity2svals, fake_entity_manager, real_max_dist, self.min_size)

        logger.debug("num removed entities: {} (reduce {} entities)".format(len(removed_entity_ids), num_removed_entities - len(removed_entity_ids)))
        cassert.test_invalid_signature_size_clusters(new_clusters,entity2svals, fake_entity_manager, self.min_signature_size)
        cassert.test_invalid_min_size_clusters(new_clusters, self.min_size)

        # raise Exception()
        new_clusters = split_big_clusters(new_clusters, pair_distance, entity2svals, fake_entity_manager, self.min_size, self.min_signature_size)
        # new_clusters = split_same_signature_clusters(new_clusters, entity2svals, fake_entity_manager, pair_distance, self.min_size, self.min_size * 2 - 1, self.min_signature_size)
        cassert.test_invalid_signature_size_clusters(new_clusters,entity2svals, fake_entity_manager, self.min_signature_size)
        cassert.test_invalid_min_size_clusters(new_clusters, self.min_size)
        # cassert.test_big_size_clusters(new_clusters, self.min_size)
        # cassert.test_big_signature_clusters(new_clusters, entity2svals, fake_entity_manager, self.min_signature_size)

        # add_fake_entities(new_clusters, entity2svals, fake_entity_manager, self.min_size, self.min_signature_size)

        return new_clusters


    def update(self, clusters, pair_distance, entity2svals, fake_entity_manager, historical_table):
        new_clusters, removed_entity_ids = build_valid_signatures_clusters_for_updation(clusters, pair_distance, entity2svals, fake_entity_manager, self.min_size, self.min_signature_size, historical_table)
        cassert.test_invalid_signature_size_clusters(new_clusters, entity2svals, fake_entity_manager, self.min_signature_size)
        cassert.test_invalid_min_size_clusters(new_clusters, self.min_size)
        cassert.test_invalid_same_signature_clusters(new_clusters, entity2svals, fake_entity_manager, historical_table)

        new_clusters = split_same_signature_clusters(new_clusters, entity2svals, fake_entity_manager, pair_distance, self.min_size, self.min_size * 2 - 1, self.min_signature_size)
        cassert.test_invalid_signature_size_clusters(new_clusters, entity2svals, fake_entity_manager, self.min_signature_size)
        cassert.test_invalid_min_size_clusters(new_clusters, self.min_size)
        cassert.test_invalid_same_signature_clusters(new_clusters, entity2svals, fake_entity_manager, historical_table)

        return new_clusters