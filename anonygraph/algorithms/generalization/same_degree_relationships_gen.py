import logging
import time
from sortedcontainers import SortedList
from functools import reduce

import anonygraph.algorithms.generalization.checkers as checkers
import anonygraph.info_loss.info as info

logger = logging.getLogger(__name__)


def _find_invalid_in_out_relation_ids(clusters, subgraph):
    result = []
    relation_ids = list(subgraph.relationship_relation_ids)

    for relation_id in relation_ids:
        start_time = time.time()
        out_invalid_clusters = []
        in_invalid_clusters = []

        for cluster in clusters:
            if checkers.is_relation_degree_invalid(
                cluster, subgraph, relation_id, 'out'
            ):
                out_invalid_clusters.append(cluster)

            if checkers.is_relation_degree_invalid(
                cluster, subgraph, relation_id, 'in'
            ):
                in_invalid_clusters.append(cluster)

        logger.debug(
            "found invalid relation_id: {} in {}".format(
                relation_id,
                time.time() - start_time
            )
        )
        logger.debug(
            "out clusters {} in clusters: {}".format(
                out_invalid_clusters, in_invalid_clusters
            )
        )

        if len(out_invalid_clusters) > 0 or len(in_invalid_clusters) > 0:
            # logger.info('invalid')
            result.append(
                (relation_id, out_invalid_clusters, in_invalid_clusters)
            )

    return result


def _init_sorted_in_out_required_user_ids(
    graph, relation_id, out_clusters, in_clusters
):
    sorted_out_entity_ids = SortedList(key=lambda item: -item[1])
    for out_cluster in out_clusters:
        generalized_out_degree = info.get_generalized_degree(
            graph, out_cluster.entity_ids, relation_id, 'out'
        )
        for out_entity_id in out_cluster:
            entity_out_degree = info.get_degree(
                graph, out_entity_id, relation_id, 'out'
            )
            if entity_out_degree != generalized_out_degree:
                sorted_out_entity_ids.add(
                    (out_entity_id, generalized_out_degree - entity_out_degree)
                )

    sorted_in_entity_ids = SortedList(key=lambda item: -item[1])
    for in_cluster in in_clusters:
        generalized_in_degree = info.get_generalized_degree(
            graph, in_cluster.entity_ids, relation_id, 'in'
        )

        for in_entity_id in in_cluster:
            entity_in_degree = info.get_degree(
                graph, in_entity_id, relation_id, 'in'
            )

            if entity_in_degree != generalized_in_degree:
                sorted_in_entity_ids.add(
                    (in_entity_id, generalized_in_degree - entity_in_degree)
                )

    return sorted_out_entity_ids, sorted_in_entity_ids


def calculate_total_required_degree(
    sorted_out_entity_ids, sorted_in_entity_ids
):
    total_out_required_degree = reduce(
        lambda x, y: x + y[1], sorted_out_entity_ids, 0
    )
    total_in_required_degree = reduce(
        lambda x, y: x + y[1], sorted_in_entity_ids, 0
    )
    return total_out_required_degree, total_in_required_degree


def _find_triplets_to_add(
    graph, relation_id, sorted_out_entity_ids, sorted_in_entity_ids
):
    out_entity_ids = [v for v, _ in sorted_out_entity_ids]
    in_entity_ids = [v for v, _ in sorted_in_entity_ids]
    # all_user_ids = list(graph.user_set)

    for i_idx, v_i in enumerate(out_entity_ids):
        for j_idx, v_j in enumerate(in_entity_ids):
            if v_i != v_j and not graph.is_edge_existed(v_i, relation_id, v_j):
                return (v_i, v_j), (i_idx, j_idx)

    return None, None


def _perform_triplet_addition(graph, relation_id, triplet):
    if triplet is None:
        return 0

    # logger.info('add edge {}'.format(triplet))

    graph.add_relationship_edge_from_id(triplet[0], relation_id, triplet[1])

    return 1


def _update_triplet_adition(ids, sorted_out_user_ids, sorted_in_user_ids):
    # vi_id = ids[0]
    # vj_id = ids[1]
    logger.debug(
        "ids: {} - out: {} - in: {}".format(
            ids, sorted_out_user_ids._lists, sorted_in_user_ids._lists
        )
    )
    if ids[0] is not None:
        out_vi = sorted_out_user_ids.pop(ids[0])

        if out_vi[1] > 1:
            new_out_vi = (out_vi[0], out_vi[1] - 1)
            sorted_out_user_ids.add(new_out_vi)

    if ids[1] is not None:
        in_vj = sorted_in_user_ids.pop(ids[1])

        if in_vj[1] > 1:
            new_in_vj = (in_vj[0], in_vj[1] - 1)
            sorted_in_user_ids.add(new_in_vj)


class FakeRelationshipAddition:
    def __call__(self, clusters, subgraph):
        num_addition = 0
        invalid_relation_ids = _find_invalid_in_out_relation_ids(
            clusters, subgraph
        )
        step = 0

        for (
            relation_id,
            out_invalid_clusters,
            in_invalid_clusters,
        ) in invalid_relation_ids:
            sorted_out_user_ids, sorted_in_user_ids = _init_sorted_in_out_required_user_ids(
                subgraph, relation_id, out_invalid_clusters, in_invalid_clusters
            )
            sorted_out_user_ids, sorted_in_user_ids = _init_sorted_in_out_required_user_ids(
                subgraph, relation_id, out_invalid_clusters, in_invalid_clusters
            )

            logger.debug(
                "relation: {} out: {} - in: {}".format(
                    relation_id, len(sorted_out_user_ids),
                    len(sorted_in_user_ids)
                )
            )
            while True:
                step += 1
                total_required_out, total_required_in = calculate_total_required_degree(
                    sorted_out_user_ids, sorted_in_user_ids
                )
                logger.debug(
                    '[step: {}] - required (out: {} - in: {}) - added: {}'.
                    format(
                        step, total_required_out, total_required_in,
                        num_addition
                    )
                )

                edge, ids = _find_triplets_to_add(
                    subgraph, relation_id, sorted_out_user_ids,
                    sorted_in_user_ids
                )

                if _perform_triplet_addition(subgraph, relation_id, edge) == 1:
                    num_addition += 1
                    _update_triplet_adition(
                        ids, sorted_out_user_ids, sorted_in_user_ids
                    )
                    calculate_total_required_degree(
                        sorted_out_user_ids, sorted_in_user_ids
                    )

                    continue

                break

        logger.info("added {} relationships edges".format(num_addition))


def _init_sorted_in_out_required_clusters(
    graph, relation_id, out_clusters, in_clusters
):
    sorted_out_user_ids = SortedList(key=lambda item: -item[1])
    for out_cluster in out_clusters:
        union_degree = info.get_generalized_degree(
            graph, out_cluster.entity_ids, relation_id, "out"
        )

        total_required_edges = 0
        for out_user_id in out_cluster.entity_ids:
            user_degree = info.get_degree(
                graph, out_user_id, relation_id, "out"
            )

            total_required_edges += union_degree - user_degree
        sorted_out_user_ids.add((out_cluster, total_required_edges))

    sorted_in_user_ids = SortedList(key=lambda item: -item[1])
    for in_cluster in in_clusters:
        union_degree = info.get_generalized_degree(
            graph, in_cluster.entity_ids, relation_id, "in"
        )

        total_required_edges = 0
        for in_user_id in in_cluster.entity_ids:
            user_degree = info.get_degree(graph, in_user_id, relation_id, "in")

            total_required_edges += union_degree - user_degree

        sorted_in_user_ids.add((in_cluster, total_required_edges))

    return sorted_out_user_ids, sorted_in_user_ids


def _find_out_connected_user_ids(graph, relation_id, user_id, user_ids):
    if user_ids is not None:
        for user2_id in user_ids:
            if graph.is_edge_existed(user_id, relation_id, user2_id):
                return user2_id
    else:
        for _, current_relation_id, user2_id in graph.get_out_edges_iter(
            user_id
        ):
            if current_relation_id == relation_id:
                return user2_id

    return None


def _find_in_connected_user_ids(graph, relation_id, user_id, user_ids):
    if user_ids is not None:
        for user2_id in user_ids:
            if graph.is_edge_existed(user2_id, relation_id, user_id):
                return user2_id
    else:
        for user2_id, current_relation_id, _ in graph.get_in_edges_iter(
            user_id
        ):
            if current_relation_id == relation_id:
                return user2_id

    return None


class DegreeDecrement:
    def __call__(self, clusters, subgraph):
        count = 0
        logger.info("start fixing invalid clusters by decreasing degree")

        invalid_relation_ids = _find_invalid_in_out_relation_ids(
            clusters, subgraph
        )
        # all_user_ids = graph.user_ids

        for (
            relation_id,
            out_invalid_clusters,
            in_invalid_clusters,
        ) in invalid_relation_ids:
            sorted_out_user_ids, sorted_in_user_ids = _init_sorted_in_out_required_clusters(
                subgraph, relation_id, out_invalid_clusters, in_invalid_clusters
            )

            out_user_ids = set()
            for cluster in out_invalid_clusters:
                out_user_ids.update(cluster.entity_ids)

            in_user_ids = set()
            for cluster in in_invalid_clusters:
                in_user_ids.update(cluster.entity_ids)

            if len(sorted_out_user_ids) > 0:
                max_out_required_edges = sorted_out_user_ids[0][1]
            else:
                max_out_required_edges = 0

            if len(sorted_in_user_ids) > 0:
                max_in_required_edges = sorted_in_user_ids[0][1]
            else:
                max_in_required_edges = 0

            logger.debug(
                "max out: {} - max in: {}".format(
                    max_out_required_edges, max_in_required_edges
                )
            )
            if (
                max_out_required_edges > max_in_required_edges and
                max_out_required_edges > 0
            ):
                # degree out degree of the maximum
                logger.info("start removing out")
                remove_cluster = sorted_out_user_ids[0][0]
                max_degree = -1
                user_degree_dict = {}
                for user_id in remove_cluster.entity_ids:
                    user_degree = info.get_degree(
                        subgraph, user_id, relation_id, 'out'
                    )
                    user_degree_dict[user_id] = user_degree

                    if user_degree > max_degree:
                        max_degree = user_degree

                for user_id, user_degree in user_degree_dict.items():
                    if user_degree == max_degree:
                        user2_id = _find_out_connected_user_ids(
                            subgraph, relation_id, user_id, in_user_ids
                        )

                        if user2_id is None:
                            user2_id = _find_out_connected_user_ids(
                                subgraph, relation_id, user_id, None
                            )

                        subgraph.remove_edge_from_id(
                            user_id, relation_id, user2_id
                        )
                        # logger.info('remove {}'.format((user_id, relation_id, user2_id)))
                        count += 1
            elif max_in_required_edges > 0:
                # degree in degree of the maximum
                logger.info("start removing in")
                remove_cluster = sorted_in_user_ids[0][0]
                max_degree = -1
                user_degree_dict = {}
                for user_id in remove_cluster.entity_ids:
                    user_degree = info.get_degree(
                        subgraph, user_id, relation_id, 'in'
                    )
                    user_degree_dict[user_id] = user_degree

                    if user_degree > max_degree:
                        max_degree = user_degree

                for user_id, user_degree in user_degree_dict.items():
                    if user_degree == max_degree:
                        user2_id = _find_in_connected_user_ids(
                            subgraph, relation_id, user_id, out_user_ids
                        )

                        if user2_id is None:
                            user2_id = _find_in_connected_user_ids(
                                subgraph, relation_id, user_id, None
                            )

                        subgraph.remove_edge_from_id(
                            user2_id, relation_id, user_id
                        )
                        # logger.info('remove {}'.format((user2_id, relation_id, user_id)))
                        count += 1
            # print('in: {}'.format(sorted_in_user_ids._lists))
            # print('out: {}'.format(sorted_out_user_ids._lists))
            # raise Exception()
            # invalid_edge = _find_invalid_edge(graph,
            #     relation_id, out_invalid_clusters, in_invalid_clusters)
            # logger.info('invalid edge: {}'.format(invalid_edge))
            # # raise Exception()
            # # TODO remove edge

            # # if invalid_edge is None:
            # #     continue

            # count += 1
            # graph.remove_edge_from_id(*invalid_edge)

        logger.info("removed: {} edges".format(count))


def _find_invalid_clusters(clusters, subgraph):
    invalid_clusters = []

    for cluster in clusters:
        if checkers.is_out_in_degree_invalid(cluster, subgraph):
            invalid_clusters.append(cluster)

    return invalid_clusters


class SameDegreeRelationshipsGenerator:
    def __init__(self):
        self.__addition_fixer = FakeRelationshipAddition()
        self.__removal_fixer = DegreeDecrement()

    def __call__(self, clusters, subgraph):
        logger.info('start generalizing degree mode')
        # keep adding
        invalid_clusters = _find_invalid_clusters(clusters, subgraph)
        step = 0
        logger.info(
            '[step: {}] number of invalid clusters: {}'.format(
                step, len(invalid_clusters)
            )
        )

        while len(invalid_clusters) > 0:
            logger.debug('graph: {}'.format(subgraph))
            step += 1
            self.__addition_fixer(clusters, subgraph)

            invalid_clusters = _find_invalid_clusters(clusters, subgraph)
            logger.info(
                '[step: {}] number of invalid clusters after adding: {}'.format(
                    step, len(invalid_clusters)
                )
            )

            if len(invalid_clusters) > 0:
                self.__removal_fixer(clusters, subgraph)
