from anonygraph.algorithms.fake_entity_manager import FakeEntityManager
import sys
import logging
logger = logging.getLogger(__name__)
import math
import numpy as np
import itertools
from numpy.core.numerictypes import sctype2char

from anonygraph.algorithms import PairDistance, Cluster, PairDistance
from anonygraph.algorithms.clustering.enforcers import split_overlap_assignmnet_enforcer as soa
from anonygraph.algorithms.clustering.enforcers import greedy_split_enforcer as gse
from anonygraph.info_loss import info


def generate_pair_distance(num_users):
    pair_distance = PairDistance()

    for entity1_id, entity2_id in itertools.combinations(range(num_users), r=2):
        distance = np.random.rand()
        pair_distance.add(entity1_id, entity2_id, distance)

    return pair_distance

def generate_s2entities_list(num_users, signature_list):
    s2entities_list = [[] for _ in range(len(signature_list))]

    for entity_id in range(num_users):
        sval_id = np.random.choice(signature_list)
        sval_idx = signature_list.index(sval_id)
        s2entities_list[sval_idx].append(entity_id)

    return s2entities_list

def generate_s2freq_list(s2entities_list):
    s2freq_list = [[] for _ in range(len(s2entities_list))]

    for sidx, entities_list in enumerate(s2entities_list):
        s2freq_list[sidx] = len(entities_list)

    return s2freq_list


def setup_logger(level):
    logger = logging.getLogger("")
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(name)-12s: %(funcName)s(%(lineno)d) %(levelname)-8s %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def generate_entity2svals(entity_ids, sval_ids):
    entity2svals = {}
    for entity_id in entity_ids:
        val_id = np.random.choice(sval_ids)
        entity2svals[entity_id] = {val_id}

    return entity2svals

def test_get_cluster_freq_stats_list(num_users, num_sensitive_vals):
    entity_ids = list(range(num_users))
    sval_ids = list(range(num_sensitive_vals))

    entity2svals = generate_entity2svals(entity_ids, sval_ids)
    signature_list = list(info.get_generalized_signature_info_from_dict(entity2svals, None, entity_ids))


    logger.debug("entity2svals: {}".format(entity2svals))
    logger.debug("signature: {}".format(signature_list))
    # generate_s2entities_list(num_users, signature_list)

    s2entities, s2num_entities = soa.get_cluster_freq_stats_list(entity_ids, entity2svals, signature_list)

    logger.debug("s2entities: {}".format(s2entities))
    logger.debug("s2num_entities: {}".format(s2num_entities))

    for sidx, entities in enumerate(s2entities):
        num_entities = s2num_entities[sidx]
        assert(num_entities == len(entities))

        sid = signature_list[sidx]

        for entity_id in entities:
            assert sid in entity2svals[entity_id]

def test_generate_s2combinations(num_users, num_sensitive_vals):
    entity_ids = list(range(num_users))
    sval_ids = list(range(num_sensitive_vals))

    entity2svals = generate_entity2svals(entity_ids, sval_ids)
    signature_list = list(info.get_generalized_signature_info_from_dict(entity2svals, None, entity_ids))

    s2entities, s2num_entities = soa.get_cluster_freq_stats_list(entity_ids, entity2svals, signature_list)

    logger.debug("s2entities: {}".format(s2entities))
    logger.debug("s2num_entities: {}".format(s2num_entities))

    for num_clusters in range(1, num_users):
        s2com = soa.generate_s2combinations(s2freq=s2num_entities, num_clusters=num_clusters)
        logger.debug("s2com: {}".format(s2com))

        for sidx, com_list in enumerate(s2com):

            num_users = s2num_entities[sidx]

            for com in com_list:
                assert(sum(com) == num_users)
                assert(len(com) == num_clusters)

                logger.debug("sidx: {} - num_users: {} - com: {}".format(sidx, num_users, com))

def test_check_valid_s2cfreq():
    s2cfreq = [[1, 1, 0], [3, 1, 0], [1, 0, 1], [1, 1, 1]]
    assert soa.check_valid_s2cfreq(s2cfreq, 2, 2) == True
    assert soa.check_valid_s2cfreq(s2cfreq, 3, 2) == False
    assert soa.check_valid_s2cfreq(s2cfreq, 2, 3) == False
    assert s2cfreq[1][0] == 3

def test_generate_valid_s2cfreq_re(num_users, num_sensitive_vals, min_size, min_signature_size):
    entity_ids = list(range(num_users))
    sval_ids = list(range(num_sensitive_vals))

    entity2svals = generate_entity2svals(entity_ids, sval_ids)
    signature_list = list(info.get_generalized_signature_info_from_dict(entity2svals, None, entity_ids))

    s2entities, s2num_entities = soa.get_cluster_freq_stats_list(entity_ids, entity2svals, signature_list)

    logger.debug("s2entities: {}".format(s2entities))
    logger.debug("s2num_entities: {}".format(s2num_entities))

    # for num_clusters in range(1, num_users):
    num_clusters = 2
    s2com = [
        [[1, 1, 1], [0, 1, 2], [0, 0, 3]],
        [[0, 1, 1], [0, 0, 2]],
        [[0, 0, 4], [1, 1, 2], [0, 1, 3]],
        [[0, 1, 1], [0, 0, 2]],
    ]


    # s2com = soa.generate_s2combinations(s2freq=s2num_entities, num_clusters=num_clusters)

    logger.debug("s2com: {}".format(s2com))

    num_all_permutations = 1
    for com_list in s2com:
        current_num_per = 0
        for com in com_list:
            current_permutations = list(itertools.permutations(com))
            current_num_per += len(current_permutations)

        num_all_permutations *= current_num_per
        logger.debug("com: {} - num permutations: {}".format(com_list, current_num_per))
        # num_all_permutations *= len()
    logger.debug("num of all permutations: {}".format(num_all_permutations))

    s2cfreqs = []
    soa.generate_valid_s2cfreq_re(s2com, min_size, min_signature_size, 0, [], s2cfreqs)

    logger.debug("num clusters: {} - s2com: {} - num of s2cfreqs: {}".format(num_clusters, s2com, len(s2cfreqs)))

def test_generate_c2entities():
    num_users = 11
    num_signatures = 4
    min_size = 2
    min_signature_size = 2
    num_clusters = 3
    entity_ids = list(range(num_users))
    sval_ids = list(range(num_signatures))

    entity2svals = generate_entity2svals(entity_ids, sval_ids)
    pair_distance = generate_pair_distance(num_users)
    signature_list = list(info.get_generalized_signature_info_from_dict(entity2svals, None, entity_ids))
    # s2entities, s2num_entities = soa.get_cluster_freq_stats_list(entity_ids, entity2svals, signature_list)

    s2entities = [
        [0, 1],
        [2, 3, 4, 5],
        [6, 7],
        [8, 9, 10]
    ]

    s2cfreq = [[1, 1, 0], [3, 1, 0], [1, 0, 1], [1, 1, 1]]
    e2sidx = {
        0:0,
        1:0,
        2:1,
        3:1,
        4:1,
        5:1,
        6:2,
        7:2,
        8:3,
        9:3,
        10:3,
    }
    soa.generate_c2entities_optimal(s2cfreq, pair_distance, s2entities, e2sidx, min_size)

def test_split_cluster():
    num_users = 11
    num_signatures = 4
    min_size = 2
    min_signature_size = 2
    entity_ids = list(range(num_users))
    sval_ids = list(range(num_signatures))

    entity2svals = generate_entity2svals(entity_ids, sval_ids)
    pair_distance = generate_pair_distance(num_users)
    signature_list = list(info.get_generalized_signature_info_from_dict(entity2svals, None, entity_ids))

    current_clusters = soa.split_cluster(
                entity_ids, signature_list, pair_distance, entity2svals,
                None, min_size, min_signature_size
            )


def test_split_cluster_beam_search():
    num_users = 200
    num_signatures = 4
    min_size = 2
    min_signature_size = 2
    entity_ids = list(range(num_users))
    sval_ids = list(range(num_signatures))

    entity2svals = generate_entity2svals(entity_ids, sval_ids)
    pair_distance = generate_pair_distance(num_users)
    signature_list = list(info.get_generalized_signature_info_from_dict(entity2svals, None, entity_ids))

    current_clusters = soa.split_cluster_heuristic(
                entity_ids, signature_list, pair_distance, entity2svals,
                None, min_size, min_signature_size, 2
            )

def generate_entity2svals_with_constraint(entity_ids, sval_ids, min_signature_size):
    entity2svals = {}

    current_signature = set()

    for entity_id in entity_ids:

        if len(current_signature) < min_signature_size:
            while(True):
                sval_id = np.random.choice(sval_ids)

                if sval_id not in current_signature:
                    break

        else:
            sval_id = np.random.choice(sval_ids)

        entity2svals[entity_id] = {sval_id}
        current_signature.add(sval_id)

    return entity2svals


def test_split_greedy(num_users, num_svals, min_size, min_signature_size):
    user_ids = list(range(num_users))
    sval_ids = list(range(num_svals))

    entity2svals = generate_entity2svals_with_constraint(user_ids, sval_ids, min_signature_size)
    logger.debug("entity2svals: {}".format(entity2svals))
    pair_distance = generate_pair_distance(num_users)

    signature = set()
    for sval_ids in entity2svals.values():
        signature.update(sval_ids)

    logger.debug("signature: {}".format(signature))
    logger.debug("min_signature_size: {} - min_size: {}".format(min_signature_size, min_size))
    cluster = Cluster.from_iter(user_ids)


    gse.split_cluster_greedy(cluster, signature, pair_distance, entity2svals, None, min_size, min_signature_size)

def test_find_valid_cluster_assignment():
    num_users = 10
    num_svals = 2
    min_size = 4
    min_signature_size = 2

    user_ids = list(range(num_users))
    sval_ids = list(range(num_svals))

    entity2svals = generate_entity2svals_with_constraint(user_ids, sval_ids, min_signature_size)
    logger.debug("entity2svals: {}".format(entity2svals))
    pair_distance = generate_pair_distance(num_users)

    signature = set()
    for sval_ids in entity2svals.values():
        signature.update(sval_ids)

    # gse.assign_valid_cluster_or_add_fake_entities(valid_clusters, invalid_clusters, pair_distance, entity2svals, fake_entity_manager, min_size, min_signature_size, max_dist)
    invalid_cluster = set()
    # valid_clusters =
    gse.find_valid_cluster_assignment(invalid_cluster, valid_clusters, pair_distance, max_dist)


def main():
    setup_logger(logging.DEBUG)

    num_users = 14
    num_sensitive_vals = 3
    min_size = 4
    min_signature_size = 3
    # signature_list = [0, 1, 2]

    # pair_distance = generate_pair_distance(num_users)
    # s2entities_list = generate_s2entities_list(num_users, signature_list)

    # logger.debug("s2entities: {}".format(s2entities_list))
    # logger.debug("pair distance: {}".format(pair_distance.to_distance_matrix()))

    # test_get_cluster_freq_stats_list(num_users, num_sensitive_vals)
    # test_generate_s2combinations(num_users, num_sensitive_vals)
    # test_check_valid_cfreq()
    # test_generate_valid_s2cfreq_re(num_users, num_sensitive_vals, min_size, min_signature_size)
    # test_generate_c2entities()
    # test_split_cluster_beam_search()
    # generate_c2freq(s2freq_arr, min_size)
    # test_split_greedy(num_users, num_sensitive_vals, min_size, min_signature_size)
    test_assign_valid_cluster_or_add_fake_entities()

if __name__ == "__main__":
    main()