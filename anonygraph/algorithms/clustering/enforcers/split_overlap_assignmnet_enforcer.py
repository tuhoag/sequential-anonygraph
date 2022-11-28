from sortedcontainers import SortedSet, SortedList
import logging
import math
import itertools
import sys
import copy
import time
import numpy as np

from sortedcontainers.sorteddict import SortedDict

from anonygraph.algorithms import Cluster
from anonygraph.assertions import clusters_assertions as cassert
from anonygraph.info_loss import info
from .base_enforcer import BaseEnforcer
from .merge_split_assignment_enforcer import calculate_real_max_dist, remove_invalid_size_and_signature_clusters, assign_valid_clusters, get_cluster_status_str, get_clusters_freq_stats, build_valid_signatures_clusters_for_updation, split_same_signature_clusters

logger = logging.getLogger(__name__)


def generate_combinations_re(n, max_size, current_arr, current_sum, result):
    # logger.debug("{},{},{},{},{}".format(n, max_size, current_arr, current_sum, result))
    if len(current_arr) > max_size or current_sum > n:
        # print("return due to oversize of over sum")
        return

    if current_sum == n and len(current_arr) <= max_size:
        for _ in range(max_size - len(current_arr)):
            current_arr.append(0)

        result.append(current_arr)
        # print("found")
        return

    if len(current_arr) == 0:
        pre_number = 1
    else:
        pre_number = current_arr[-1]

    for num in range(pre_number, n + 1):
        generate_combinations_re(
            n, max_size, current_arr + [num], current_sum + num, result
        )


def generate_combinations(n, size):
    """Generate combinations of size numbers whose sum is equal to n

    Args:
        n ([type]): [description]
        size ([type]): [description]

    Returns:
        [type]: [description]
    """
    logger.debug("generating combination: sum={}, size={}".format(n, size))
    result = []
    start_time = time.time()
    generate_combinations_re(n, size, [], 0, result)
    logger.debug("finished generating combinations in {}".format(time.time() - start_time))
    return result


def generate_s2combinations(s2freq, num_clusters):
    """For every sensitive value s_i in s2freq, generating all combinations that have 'num_clusters' numbers and sum of these numbers are equal to s2freq[s_i]

    Args:
        s2freq (list): list where each item stores the number of entities whose sensitive value is equal to the sensitive value of sidx (which can be selected in signature_list).
        num_clusters (int): number of clusters. This is the length of each combination.

    Returns:
        list: each item at index 'sidx' is a list of all combinations of sensitive value at 'sidx'.
    """
    # result = [[]] * len(s2freq)
    result = [None for _ in range(len(s2freq))]
    # print(len(result), len(s2freq))
    # print(result)
    for sidx in range(len(s2freq)):
        # print("sid=", sidx)
        combinations = generate_combinations(s2freq[sidx], num_clusters)

        # print("before: ", result)
        # result[sid].extend(combinations)
        result[sidx] = combinations
        # print("after: ", result)

    return result


def get_cluster_freq_stats_list(cluster, entity2svals, signature_list):
    s2entities = [[] for _ in range(len(signature_list))]
    s2num_entities = [0 for _ in range(len(signature_list))]
    e2sidx = {}

    for entity_id in cluster:
        sval_id = list(entity2svals[entity_id])[0]
        sval_idx = signature_list.index(sval_id)

        s2entities[sval_idx].append(entity_id)
        s2num_entities[sval_idx] += 1

        e2sidx[entity_id] = sval_idx

    return s2entities, s2num_entities, e2sidx


def check_valid_s2cfreq(cfreq, min_size, min_signature_size):
    """Check if a frequency assignment is valid.

    Args:
        cfreq (list(list, list)): a frequency assignment. First dimension is sensitive values, 2nd one is frequency of clusters for corresponding sensitive value. For instance, cfreq[0][1] is the num of entities having sidx 0 in cluster 1.
        min_size ([type]): [description]
        min_signature_size ([type]): [description]

    Returns:
        [type]: [description]
    """
    logger.debug("checking cfreq: {}".format(cfreq))
    num_clusters = len(cfreq[0])

    # for _ in cfreq:

    # check k valid

    # logger.debug(num_clusters)
    # raise Exception()
    for cidx in range(num_clusters):
        size = 0
        signature_size = 0

        # print(size, signature_size)
        for freq in cfreq:
            size += freq[cidx]
            signature_size += freq[cidx] != 0

            # print(size, signature_size)

        logger.debug(
            "cid: {} - size: {} - signature size: {}".format(
                cidx, size, signature_size
            )
        )
        if 0 < size < min_size:
            logger.debug(
                "cluster {} (size: {}) is invalid min size ({})".format(
                    cidx, size, min_size
                )
            )

            # raise Exception()
            return False

        if 0 < signature_size < min_signature_size:
            logger.debug(
                "cluster {} (sig size: {}) is invalid min sig size ({})".format(
                    cidx, signature_size, min_signature_size
                )
            )

            # raise Exception()
            return False

    return True


def generate_valid_s2cfreq_re(
    s2com, min_size, min_signature_size, sidx, current_combination, result
):
    """[summary]

    Args:
        s2com ([type]): [description]
        min_size ([type]): [description]
        min_signature_size ([type]): [description]
        sidx ([type]): [description]
        current_combination ([type]): [description]
        result ([type]): [description]
    """
    logger.debug(
        "before: sidx: {}, current: {}, result len: {}".format(
            sidx, current_combination, len(result)
        )
    )

    # print(s2com, sidx, current_combination, result)
    if sidx == len(s2com):
        # found
        # print('found')
        if check_valid_s2cfreq(
            current_combination, min_size, min_signature_size
        ):
            result.append(current_combination)

        return

    for com in s2com[sidx]:
        cluster_sfreq = set(itertools.permutations(com))
        # logger.debug("current com: {} - permutations unique {} - normal {}".format(com, perm, list(itertools.permutations(com))))

        # raise Exception()
        for sfreq in cluster_sfreq:
            generate_valid_s2cfreq_re(
                s2com, min_size, min_signature_size, sidx + 1,
                current_combination + [sfreq], result
            )

    logger.debug(
        "after: sidx: {}, current: {}, result len: {}".format(
            sidx, current_combination, len(result)
        )
    )


def generate_clusters_from_assignment(
    assignment, s2entities_list, num_clusters
):
    clusters = [[] for _ in range(num_clusters)]

    for sidx, sasignment in enumerate(assignment):
        logger.debug("sidx: {} - sassignment: {}".format(sidx, sasignment))

        for eidx, cid in enumerate(sasignment):
            # print(eidx, cid, s2entities_list[sidx][eidx])
            clusters[cid].append(s2entities_list[sidx][eidx])
            # print(clusters)

    return clusters

def initialize_s2cu(cfreq, s2entities_list):
    num_svals = len(s2entities_list)
    num_clusters = len(cfreq[0])

    initial_s2cu = [
        [set() for _ in range(num_clusters)] for _ in range(len(cfreq))
    ]


    # logger.debug(initial_s2cu)
    for sidx in range(num_svals):
        entities = s2entities_list[sidx]
        freq = cfreq[sidx]

        # logger.debug(
        #     "sidx={}, entities={}, freq={}".format(sidx, entities, freq)
        # )

        eidx = 0
        for cidx, count in enumerate(freq):
            initial_s2cu[sidx]
            for _ in range(count):
                # logger.debug("cidx={}, eidx={}".format(cidx, eidx))
                initial_s2cu[sidx][cidx].add(entities[eidx])
                eidx += 1

    return initial_s2cu

def initialize_c2edges(cfreq, s2entities_list, initial_s2cu, pair_distance):
    num_svals = len(s2entities_list)
    num_clusters = len(cfreq[0])

    initial_c2edges = [SortedSet(key=lambda item: -item[0]) for _ in range(num_clusters)]

    for cidx in range(num_clusters):
        for s1idx in range(num_svals):
            for e1id in initial_s2cu[s1idx][cidx]:
                for s2idx in range(num_svals):
                    for e2id in initial_s2cu[s2idx][cidx]:
                        dist = pair_distance.get_distance(e1id, e2id)

                        if e1id == e2id:
                            continue
                        if e1id <= e2id:
                            key = (dist, e1id, e2id)
                        else:
                            key = (dist, e2id, e1id)

                        initial_c2edges[cidx].add(key)

    return initial_c2edges

def get_c2longest_edge(cfreq, s2entities_list, s2cu, pair_distance):
    num_svals = len(s2entities_list)
    num_clusters = len(cfreq[0])

    c2longest_edge = [(-sys.maxsize, None, None) for _ in range(num_clusters)]

    for cidx in range(num_clusters):
        for s1idx in range(num_svals):
            for e1id in s2cu[s1idx][cidx]:
                for s2idx in range(num_svals):
                    for e2id in s2cu[s2idx][cidx]:
                        dist = pair_distance.get_distance(e1id, e2id)

                        if e1id == e2id:
                            continue

                        if dist > c2longest_edge[cidx][0]:
                            if e1id <= e2id:
                                key = (dist, e1id, e2id)
                            else:
                                key = (dist, e2id, e1id)

                            c2longest_edge[cidx] = key

    return c2longest_edge



class ClustersAssignmentState():
    def __init__(self):
        self.__s2cu = None
        # self.__c2edges = None
        self.__c2longest_edge = None
        self.__cost = None
        self.__hash = None

    @staticmethod
    def initialize(cfreq, s2entities_list, pair_distance):
        new_state = ClustersAssignmentState()
        new_state.cfreq = cfreq
        new_state.s2entities_list = s2entities_list
        new_state.pair_distance = pair_distance

        new_state.__s2cu = initialize_s2cu(cfreq, s2entities_list)
        # new_state.__c2edges = initialize_c2edges(cfreq, s2entities_list, new_state.__s2cu, pair_distance)
        new_state.__c2longest_edge = get_c2longest_edge(cfreq, s2entities_list, new_state.__s2cu, pair_distance)

        return new_state

    def __str__(self):
        return "s2cu: {} - c2ledge: {}".format(self.__s2cu, self.__c2longest_edge)

    def __repr__(self):
        return str(self)


    @property
    def cost(self):
        if self.__cost is None:
            self.__cost = ClustersAssignmentState.calculate_cost(self)

        return self.__cost

    @staticmethod
    def calculate_cost(state):
        # logger.debug("s2cu: {} - c2ledge: {}".format(state.__s2cu, state.__c2longest_edge))

        cost = 0
        for cidx, edge in enumerate(state.__c2longest_edge):
            max_dist = edge[0]
            num_entities = sum(map(lambda item: len(item[cidx]), state.__s2cu))
            entities = list(map(lambda item: item[cidx], state.__s2cu))
            # logger.debug("entities: {}".format(entities))
            # logger.debug("cidx: {} - max: {} - num entities: {} - {}".format(cidx, max_dist, num_entities, max_dist * num_entities))

            # if num_entities == 0:
            #     raise Exception()
            if num_entities > 1:
                cost += max_dist * num_entities

            # logger.debug("cost: {}".format(cost))
        # raise Exception()
        if cost < 0:
            raise Exception()
        return cost

    def copy(self, s2cu, c2longhest_edge):
        new_state = ClustersAssignmentState()
        new_state.cfreq = self.cfreq
        new_state.s2entities_list = self.s2entities_list
        new_state.pair_distance = self.pair_distance

        new_state.__s2cu = s2cu
        new_state.__c2longest_edge = c2longhest_edge

        return new_state


    def generate_by_swap(self, sidx, cidx1, entity1_id, cidx2, entity2_id):
        new_s2cu = copy.deepcopy(self.__s2cu)

        new_s2cu[sidx][cidx1].remove(entity1_id)
        new_s2cu[sidx][cidx2].remove(entity2_id)

        new_s2cu[sidx][cidx1].add(entity2_id)
        new_s2cu[sidx][cidx2].add(entity1_id)

        # logger.debug(self.__s2cu[sidx])
        # logger.debug(new_s2cu[sidx])

        new_c2longest_edge = get_c2longest_edge(self.cfreq, self.s2entities_list, new_s2cu, self.pair_distance)

        return self.copy(new_s2cu, new_c2longest_edge)

    def find_next_states(self, e2sidx):
        # for each cluster, choose the highest cost cluster and generate it next state by swaping nodes envoling in the highest edge.

        next_states = []

        for cidx1, edge in enumerate(self.__c2longest_edge):
            # longest_edge1 = edge[0]
            logger.debug("cidx: {} - longest: {}".format(cidx1, edge))

            for entity1_id in [edge[1], edge[2]]:
            # find its sidx
                if entity1_id is None:
                    continue

                # logger.debug("{}, {}".format(e2sidx, entity1_id))
                sidx = e2sidx[entity1_id]

                # swap two users with other clusters
                # logger.debug(sidx)
                # logger.debug(self.__s2cu[sidx])
                for cidx2, entities2 in enumerate(self.__s2cu[sidx]):
                    if cidx2 == cidx1:
                        continue

                    # logger.debug("cidx2: {} - entities: {}".format(cidx2, entities2))

                    for entity2_id in entities2:
                        # logger.debug("swap entity id: {} (cidx: {}) with entity id: {} (cidx: {})".format(entity1_id, cidx1, entity2_id, cidx2))

                        next_state = self.generate_by_swap(sidx, cidx1, entity1_id, cidx2, entity2_id)
                        next_states.append(next_state)
        # logger.debug("found {} next states".format(len(next_states)))
        # raise Exception()
        return next_states

    def get_code(self):
        if self.__hash is None:
            # calculate
            result = []
            for c2u in self.__s2cu:

                for entities in c2u:
                    result.append(tuple(entities))

                # logger.debug(tuple(c2u))

            # logger.debug("{} -> {}".format(result, tuple(result)))
            self.__hash = tuple(result)
            # logger.debug(tuple(self.__s2cu))
            # raise Exception()
            # pass

        return self.__hash

    def generate_clusters(self):
        num_clusters = len(self.__c2longest_edge)

        clusters = [[] for _ in range(num_clusters)]

        for c2u_list in self.__s2cu:
            for cidx in range(num_clusters):
                clusters[cidx].extend(c2u_list[cidx])

        final_clusters = []
        for cluster in clusters:
            if len(cluster) == 0:
                continue

            final_clusters.append(Cluster.from_iter(cluster))

        logger.debug("s2cu: {}".format(self.__s2cu))
        logger.debug("clusters: {}".format(clusters))


        if len(final_clusters) <= 1:
            return None

        # raise Exception()
        return final_clusters

def generate_c2entities_greedy(cfreq, pair_distance, s2entities_list, e2sidx, min_size):
    logger.debug("current cfreq: {}".format(cfreq))
    if cfreq is None:
        return sys.maxsize, None

    num_svals = len(s2entities_list)
    num_clusters = len(cfreq[0])

    # create the initial c2entities assignment
    initial_state = ClustersAssignmentState.initialize(cfreq, s2entities_list, pair_distance)
    initial_cost = initial_state.cost

    logger.debug("initial state: {}".format(initial_state))

    seen_states = {initial_state.get_code()}

    recent_scores = []

    current_state = (initial_cost, initial_state) #cost, s2cu, c2edges
    is_updated = True
    while(is_updated):
        logger.debug("seen: {} - current_best: {}".format(len(seen_states), current_state))

        next_states = current_state[1].find_next_states(e2sidx)

        is_updated = False
        for next_state in next_states:
            if next_state.get_code() not in seen_states:
                seen_states.add(next_state.get_code())

                if next_state.cost < current_state[0]:
                    current_state = (next_state.cost, next_state)
                    recent_scores.append(current_state[0])

                    is_updated = True

        # logger.debug("closed: {}".format(seen_states))
        # logger.debug("open: {}".format(open_states))

    logger.debug("best: {}".format(current_state))
    logger.debug("updated {} times with recent scores are: {}".format(len(recent_scores), recent_scores))
    # raise Exception()
    return current_state


def generate_c2entities_optimal(
    cfreq, pair_distance, s2entities_list, e2sidx, min_size
):
    logger.debug("current cfreq: {}".format(cfreq))
    num_svals = len(s2entities_list)
    num_clusters = len(cfreq[0])

    # create the initial c2entities assignment
    initial_state = ClustersAssignmentState.initialize(cfreq, s2entities_list, pair_distance)
    initial_cost = initial_state.cost

    logger.debug("initial state: {}".format(initial_state))

    open_states = SortedSet(key=lambda item: item[0])
    open_states.add((initial_cost, initial_state))
    seen_states = {initial_state.get_code()}

    logger.debug("open: {}".format(open_states))
    recent_scores = []
    best_state = (sys.maxsize, None) #cost, s2cu, c2edges
    while(len(open_states) > 0):
        logger.debug("open: {} - seen: {} - current_best: {}".format(len(open_states), len(seen_states), best_state))
        current_cost, current_state = open_states.pop(0)
        # seen_states.add(current_state.get_code())

        if current_cost < best_state[0]:
            best_state = (current_cost, current_state)
            recent_scores.append(best_state[0])

        next_states = current_state.find_next_states(e2sidx)

        for next_state in next_states:
            if next_state.get_code() not in seen_states:
                seen_states.add(next_state.get_code())

                open_states.add((next_state.cost, next_state))

        # logger.debug("closed: {}".format(seen_states))
        # logger.debug("open: {}".format(open_states))

    logger.debug("best: {}".format(best_state))
    logger.debug("updated {} times with recent scores are: {}".format(len(recent_scores), recent_scores))
    # raise Exception()
    return best_state
    # return best_assignment[1]


def generate_valid_cfreq(s2com, min_size, min_signature_size):
    cluster_freq = []
    generate_valid_s2cfreq_re(
        s2com, min_size, min_signature_size, 0, [], cluster_freq
    )

    return cluster_freq


def split_cluster(
    cluster, signature, pair_distance, entity2svals, fake_entity_manager,
    min_size, min_signature_size
):
    signature_list = list(signature)
    signature_size = len(signature)
    s2entities, s2num_entities, e2sidx = get_cluster_freq_stats_list(
        cluster, entity2svals, signature_list
    )
    logger.debug("s2entities: {}".format(s2entities))
    logger.debug("s2num_entities: {}".format(s2num_entities))

    if min_signature_size == signature_size:
        num_clusters = min(s2num_entities)
    else:
        num_clusters = math.floor(len(cluster) / min_size)

    if num_clusters == 1:
        return [cluster]

    logger.debug(
        "split cluster having {} entities to {} clusters".format(
            len(cluster), num_clusters
        )
    )

    start_time = time.time()
    s2com = generate_s2combinations(s2num_entities, num_clusters)
    for sidx, coms in enumerate(s2com):
        logger.debug("sidx: {} - num of coms: {}".format(sidx, len(coms)))
    logger.info("finished generating combinations in {}".format(time.time() - start_time))

    # raise Exception()

    start_time = time.time()
    cluster_freq = []
    generate_valid_s2cfreq_re(
        s2com, min_size, min_signature_size, 0, [], cluster_freq
    )
    logger.info("finished generating {} s2cfreqs in {}".format(len(cluster_freq), time.time() - start_time))

    if len(cluster_freq) == 0:
        return None


    start_time = time.time()
    best_assignment = (sys.maxsize, None, None)
    for i in range(len(cluster_freq)):
        inner_start_time = time.time()
        score, clusters_assignment_state = generate_c2entities_greedy(
            cluster_freq[i], pair_distance, s2entities, e2sidx, min_size
        )
        logger.debug("[{}/{}]finished an interation in {}".format(i, len(cluster_freq), time.time() - inner_start_time))
        if score < best_assignment[0]:
            best_assignment = (score, cluster_freq[i], clusters_assignment_state)

        # raise Exception()
    logger.info("finished finding best assignment in {}".format(time.time() - start_time))
    logger.info("found best: {}".format(best_assignment))

    if best_assignment[2] is None:
        raise Exception()

    new_clusters = best_assignment[2].generate_clusters()

    return new_clusters

def initialize_s2cfreq(num_clusters, signature_size, s2num_entities):

    initial_s2cfreq = [[] for _ in range(signature_size)]

    for sidx in range(signature_size):
        initial_s2cfreq[sidx].append(s2num_entities[sidx])

        for _ in range(1, num_clusters):
            initial_s2cfreq[sidx].append(0)

    # c2u_state = generate_c2entities_greedy(initial_s2cfreq, )
    # logger.debug(initial_s2cfreq)
    # raise Exception()
    return initial_s2cfreq

def get_beam_list_item(beam_list, index):
    result = []
    for temp in beam_list:
        result.append(temp[index])
    return result

def generate_s2permutations(s2com):
    s2per = [[] for _ in range(len(s2com))]

    for sidx, com_list in enumerate(s2com):
        for com in com_list:
            all_permutations = set(itertools.permutations(com))
            s2per[sidx].extend(all_permutations)
        # logger.debug("sidx: {} - com_list: {}".format(sidx, com_list))


    # raise Exception()
    return s2per

def check_valid_s2cfreq_arr(s2cfreq, min_size, min_signature_size):
    # logger.debug("s2cfreq: {}".format(s2cfreq))
    for cidx in range(s2cfreq.shape[1]):
        signature_size = 0
        cluster_size = 0

        for sidx in range(s2cfreq.shape[0]):
            cluster_size += s2cfreq[sidx, cidx]
            signature_size += 1 if s2cfreq[sidx, cidx] > 0 else 0

        # logger.debug("cluster: {} - cluster size: {} - signature_size: {}".format(s2cfreq[:,cidx], cluster_size, signature_size))

        if 0 < cluster_size < min_size or 0 < signature_size < min_signature_size:
            return False

        # if cluster_size >= 2 * min_size:
        #     return False

    return True

def count_num_of_clusters(s2cfreq):
    return np.sum(np.sum(s2cfreq, axis=0) > 0)



def heuristic_finding_s2cfreq_re(s2sorted_com, s2sorted_per, sorted_sidxes, num_clusters, min_size, min_signature_size, sidx, current_s2cfreq, results):
    logger.debug("sidx: {} - result: {}".format(sidx, current_s2cfreq))
    if sidx == len(s2sorted_com):
        # it combined enough

        # s2cfreq = np.zeros((len(sorted_sidxes), num_clusters), dtype=int)
        # for real_sidx, com_idx, pidx in current_result:
        #     s2cfreq[real_sidx,:] = s2sorted_per[real_sidx][com_idx][pidx]
        #     # logger.debug("real sidx: {} - per: {}".format(real_sidx, s2sorted_per[real_sidx][com_idx][pidx]))

        num_resulting_clusters = np.sum(np.sum(current_s2cfreq, axis=0) > 0)
        # logger.debug(current_s2cfreq)
        # logger.debug(num_resulting_clusters)
        # raise Exception()

        if check_valid_s2cfreq_arr(current_s2cfreq, min_size, min_signature_size):
            # if num_resulting_clusters == num_clusters:
            return current_s2cfreq

            # results.append((num_resulting_clusters, s2cfreq))


        # this is invalid
        # logger.debug("not found")
        # raise Exception()
        return None

    real_sidx = sorted_sidxes[sidx][1]
    # logger.debug(real_sidx)
    # logger.debug(len(s2sorted_per))
    # logger.debug(len(s2sorted_per[real_sidx]))
    for com_idx, per_list in enumerate(s2sorted_per[real_sidx]):
        # sorted permutations by num of clusters it can generate
        sorted_pidxes = SortedList(key=lambda item: -item[0])
        min_num = sys.maxsize
        max_num = -sys.maxsize
        for pidx, _ in enumerate(per_list):
            new_s2cfreq = current_s2cfreq.copy()
            new_s2cfreq[real_sidx,:] = s2sorted_per[real_sidx][com_idx][pidx]

            num_resulting_clusters = count_num_of_clusters(new_s2cfreq)
            sorted_pidxes.add((num_resulting_clusters, new_s2cfreq, pidx))
            max_num = max(max_num, num_resulting_clusters)
            min_num = min(min_num, num_resulting_clusters)


        if max_num != min_num:
            for next_num_resulting_clusters, next_s2cfreq, next_pidx in sorted_pidxes:
                logger.debug("{},{},{}".format(next_num_resulting_clusters, next_s2cfreq, next_pidx))

            # raise Exception()
        # logger.debug(len(s2sorted_per[real_sidx]))
        # logger.debug(len(per_list))
        # raise Exception()

        for next_num_resulting_clusters, next_s2cfreq, next_pidx in sorted_pidxes:
            c2freq = heuristic_finding_s2cfreq_re(s2sorted_com, s2sorted_per, sorted_sidxes, num_clusters, min_size, min_signature_size, sidx + 1, next_s2cfreq, results)

            if c2freq is not None:
                return c2freq

    return None

def split_cluster_heuristic(
    cluster, signature, pair_distance, entity2svals, fake_entity_manager,
    min_size, min_signature_size, num_clusters=None):
    signature_list = list(signature)
    signature_size = len(signature)
    s2entities, s2num_entities, e2sidx = get_cluster_freq_stats_list(
        cluster, entity2svals, signature_list
    )
    logger.debug("s2entities: {}".format(s2entities))
    logger.debug("s2num_entities: {}".format(s2num_entities))

    if num_clusters is None:
        if min_signature_size == signature_size:
            num_clusters = min(s2num_entities)
        else:
            num_clusters = math.floor(len(cluster) / min_size)

    if num_clusters == 1:
        return None

    logger.debug(
        "split cluster having {} entities to {} clusters".format(
            len(cluster), num_clusters
        )
    )

    start_time = time.time()
    s2com = generate_s2combinations(s2num_entities, num_clusters)
    s2sorted_com = [SortedList() for _ in range(signature_size)]
    for sidx in range(signature_size):
        for com in s2com[sidx]:
            std_score = np.std(com)
            s2sorted_com[sidx].add((std_score, com))

    # logger.info(s2sorted_com)
    logger.info("finished generating combinations in {}".format(time.time() - start_time))

    start_time = time.time()
    s2sorted_per = [[] for _ in range(signature_size)]
    for sidx in range(signature_size):
        inner_start_time = time.time()
        for std_score, com in s2sorted_com[sidx]:
            all_permutations = list(set(itertools.permutations(com)))
            s2sorted_per[sidx].append(all_permutations)
        logger.debug("finished generating permutations for sidx {} in {}".format(sidx, time.time() - inner_start_time))

    logger.info("finished generating permutations in {}".format(time.time() - start_time))

    # logger.debug(s2sorted_per)
    # logger.debug(s2sorted_com[0][0])
    # logger.debug(s2sorted_per[0][0])

    sorted_sidxes = SortedList(key=lambda item: -item[0])
    for sidx, com_list in enumerate(s2sorted_per):
        logger.debug("sidx: {} - count: {}".format(sidx, len(com_list)))
        sorted_sidxes.add((len(com_list), sidx))

    logger.debug(sorted_sidxes)

    all_s2cfreqs = []
    initial_s2cfreq = np.zeros((signature_size, num_clusters), dtype=int)
    best_s2cfreq = heuristic_finding_s2cfreq_re(s2sorted_com, s2sorted_per, sorted_sidxes, num_clusters, min_size, min_signature_size, 0, initial_s2cfreq, all_s2cfreqs)

    if best_s2cfreq is None:
        logger.debug(s2sorted_com)
        logger.debug(s2num_entities)
        logger.debug(num_clusters)
        for cur_s2cfreq in all_s2cfreqs:
            logger.debug(cur_s2cfreq)
        # logger.debug(all_s2cfreqs)
        # raise Exception()
        return None


    score, assignment = generate_c2entities_greedy(best_s2cfreq, pair_distance, s2entities, e2sidx, min_size)

    if assignment is None:
        if best_s2cfreq is not None:
            raise Exception()
        return None

    logger.debug(best_s2cfreq)
    logger.debug(assignment)
    logger.debug(score)
    clusters = assignment.generate_clusters()

    logger.debug(clusters)
    # raise Exception()


    return clusters



def split_big_clusters(
    clusters, pair_distance, entity2svals, fake_entity_manager, min_size,
    min_signature_size
):
    new_clusters = []

    open_clusters = [cluster for cluster in clusters]

    while len(open_clusters) > 0:
        cluster = open_clusters.pop(0)

        signature = info.get_generalized_signature_info_from_dict(
            entity2svals, fake_entity_manager, cluster
        )

        cluster_size = len(cluster)
        signature_size = len(signature)

        logger.info(
            "current cluster has {} entities (remaining clusters: {})".format(len(cluster), len(open_clusters))
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
            logger.info("add cluster")
            new_clusters.append(cluster)
        else:
            current_clusters = split_cluster_heuristic(
                cluster, signature, pair_distance, entity2svals,
                fake_entity_manager, min_size, min_signature_size, 2
            )

            # cassert.test_invalid_min_size_clusters(current_clusters, min_size)
            # cassert.test_invalid_signature_size_clusters(current_clusters, entity2svals, fake_entity_manager, min_signature_size)

            logger.debug("splited clusters: {}".format(current_clusters))

            if current_clusters is None:
                logger.info("Cannot split cluster: {}".format(cluster))
                new_clusters.append(cluster)
            else:
                logger.info(
                    "split to {} clusters".format(
                        len(current_clusters)
                    )
                )
                open_clusters.extend(current_clusters)


    return new_clusters


class SplitOverlapAssignmentEnforcer(BaseEnforcer):
    def __init__(self, min_size, min_signature_size, max_dist):
        self.min_size = min_size
        self.min_signature_size = min_signature_size
        self.max_dist = max_dist

    def __call__(
        self, clusters, pair_distance, entity2svals, fake_entity_manager
    ):
        real_max_dist = calculate_real_max_dist(pair_distance, self.max_dist)
        logger.debug("real max dist: {}".format(real_max_dist))

        new_clusters, removed_entity_ids = remove_invalid_size_and_signature_clusters(
            clusters, pair_distance, entity2svals, fake_entity_manager,
            self.min_size, self.min_signature_size
        )
        # cassert.test_invalid_signature_size_clusters(new_clusters, entity2svals, fake_entity_manager, self.min_signature_size)
        # cassert.test_invalid_min_size_clusters(new_clusters, self.min_size)

        num_removed_entities = len(removed_entity_ids)
        assign_valid_clusters(
            new_clusters, removed_entity_ids, pair_distance, entity2svals,
            fake_entity_manager, real_max_dist, self.min_size
        )

        logger.debug(
            "num removed entities: {} (reduce {} entities)".format(
                len(removed_entity_ids),
                num_removed_entities - len(removed_entity_ids)
            )
        )
        # cassert.test_invalid_signature_size_clusters(new_clusters,entity2svals, fake_entity_manager, self.min_signature_size)
        # cassert.test_invalid_min_size_clusters(new_clusters, self.min_size)

        new_clusters = split_big_clusters(
            new_clusters, pair_distance, entity2svals, fake_entity_manager,
            self.min_size, self.min_signature_size
        )
        # new_clusters = split_same_signature_clusters(new_clusters, entity2svals, fake_entity_manager, pair_distance, self.min_size, self.min_size * 2 - 1, self.min_signature_size)
        # cassert.test_invalid_signature_size_clusters(new_clusters,entity2svals, fake_entity_manager, self.min_signature_size)
        # cassert.test_invalid_min_size_clusters(new_clusters, self.min_size)

        return new_clusters

    def update(
        self, clusters, pair_distance, entity2svals, fake_entity_manager,
        historical_table
    ):
        new_clusters, removed_entity_ids = build_valid_signatures_clusters_for_updation(clusters, pair_distance, entity2svals, fake_entity_manager, self.min_size, self.min_signature_size, historical_table)
        # cassert.test_invalid_signature_size_clusters(new_clusters, entity2svals, fake_entity_manager, self.min_signature_size)
        # cassert.test_invalid_min_size_clusters(new_clusters, self.min_size)
        # cassert.test_invalid_same_signature_clusters(new_clusters, entity2svals, fake_entity_manager, historical_table)

        new_clusters = split_same_signature_clusters(new_clusters, entity2svals, fake_entity_manager, pair_distance, self.min_size, self.min_size * 2 - 1, self.min_signature_size)
        # cassert.test_invalid_signature_size_clusters(new_clusters, entity2svals, fake_entity_manager, self.min_signature_size)
        # cassert.test_invalid_min_size_clusters(new_clusters, self.min_size)
        # cassert.test_invalid_same_signature_clusters(new_clusters, entity2svals, fake_entity_manager, historical_table)

        return new_clusters