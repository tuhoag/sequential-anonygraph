import logging

import anonygraph.info_loss as ifn
from anonygraph.info_loss import info
import anonygraph.utils.general as utils

from anonygraph.constants import *


logger = logging.getLogger(__name__)

def calculate_radm(clusters, subgraph, args):
    ifn_fn = ifn.AttributeOutInDegreeInfoLoss(subgraph, {"alpha_adm": 0.5, "alpha_dm": 0.5})

    result = {}
    for cluster in clusters:
        result.update(ifn_fn.calculate_for_each_entity(cluster))

    if len(result) == 0:
        score = 0
    else:
        score = sum(result.values()) / len(result)
    return score

def calculate_adm(clusters, subgraph, args):
    ifn_fn = ifn.AttributeOutInDegreeInfoLoss(subgraph, {"alpha_adm": 0.5, "alpha_dm": 0.5})

    result = {}
    for cluster in clusters:
        result.update(ifn_fn.calculate_for_each_entity(cluster))

    num_removed_entities = 0
    for entity_id in subgraph.entity_ids:
        if entity_id not in result:
            num_removed_entities += 1 # removed user

    score = (sum(result.values()) + 1 * num_removed_entities) / subgraph.num_entities

    return score

def calculate_num_fake_entities(clusters, subgraph, args):
    count = 0

    for cluster in clusters:
        for entity_id in cluster:
            if not subgraph.is_entity_id(entity_id):
                count += 1

    return count

def calculate_num_real_entities(clusters, subgraph, args):
    count = 0

    for cluster in clusters:
        for entity_id in cluster:
            if subgraph.is_entity_id(entity_id):
                count += 1


def calculate_num_real_edges(clusters, subgraph, args):
    return subgraph.num_edges

def calculate_num_fake_edges(clusters, graph, args):
    result = 0
    relation_ids = graph.relationship_relation_ids
    logger.debug("relation ids: {}".format(relation_ids))
    for cluster in clusters:
        out_union_info, out_entities_info = ifn.info.get_generalized_degree_info(graph, cluster, "out")
        in_union_info, in_entities_info = ifn.info.get_generalized_degree_info(graph, cluster, "in")

        for relation_id in relation_ids:
            out_generalized_degree = out_union_info.get(relation_id, 0)
            in_generalized_degree = in_union_info.get(relation_id, 0)

            for entity_id in cluster:
                out_entity_degree = out_entities_info.get(entity_id).get(relation_id, 0)
                in_entity_degree = in_entities_info.get(entity_id).get(relation_id, 0)

                logger.debug(out_entities_info)
                logger.debug(in_entities_info)
                # raise Exception(relation_id)

                # if out_generalized_degree is None or out_entity_degree is None:
                    # raise Exception("{}({}) {} {} {}".format(relation_id, relation_ids, out_union_info, out_generalized_degree, out_entity_degree))
                    # raise Exception("{} {} {} {}".format(out_union_info, out_entities_info, in_union_info, in_entities_info))
                out_degree_dif = out_generalized_degree - out_entity_degree
                in_degree_dif = in_generalized_degree - in_entity_degree

                generalized_degree = min(out_degree_dif, in_degree_dif)
                result += generalized_degree

    return result

def calculate_dm(clusters, subgraph, args):
    ifn_fn = ifn.OutInDegreeInfoLoss(subgraph, {"alpha_adm": 0.5, "alpha_dm": 0.5})

    result = {}
    for cluster in clusters:
        result.update(ifn_fn.calculate_for_each_entity(cluster.to_list()))

    if len(result) == 0:
        score = 0
    else:
        score = sum(result.values()) / len(result)

    return score

def calculate_odm(clusters, subgraph, args):
    ifn_fn = ifn.OutDegreeInfoLoss(subgraph, {"alpha_adm": 0.5, "alpha_dm": 0.5})

    result = {}
    for cluster in clusters:
        result.update(ifn_fn.calculate_for_each_entity(cluster.to_list()))

    if len(result) == 0:
        score = 0
    else:
        score = sum(result.values()) / len(result)

    return score

def calculate_idm(clusters, subgraph, args):
    ifn_fn = ifn.InDegreeInfoLoss(subgraph, {"alpha_adm": 0.5, "alpha_dm": 0.5})

    result = {}
    for cluster in clusters:
        result.update(ifn_fn.calculate_for_each_entity(cluster.to_list()))

    if len(result) == 0:
        score = 0
    else:
        score = sum(result.values()) / len(result)

    return score

def calculate_am(clusters, subgraph, args):
    ifn_fn = ifn.AttributeInfoLoss(subgraph, {"alpha_adm": 0.5, "alpha_dm": 0.5})

    result = {}
    for cluster in clusters:
        result.update(ifn_fn.calculate_for_each_entity(cluster.to_list()))

    if len(result) == 0:
        score = 0
    else:
        score = sum(result.values()) / len(result)

    return score

def calculate_anonymity(clusters, subgraph, args):
    clusters_sizes = [len(cluster) for cluster in clusters]

    if len(clusters_sizes) == 0:
        anonymity = 0
    else:
        anonymity = min(clusters_sizes)

    # raise Exception("clusters: {} - anonymity: {} ({})".format(clusters, anonymity, clusters_sizes))

    return anonymity

def calculate_num_big_clusters(clusters, subgraph, args):
    min_size = args["k"]
    min_signature_size = args["l"]

    logger.debug(args)
    if args["anony_mode"] == CLUSTERS_ANONYMIZATION_MODE:
        fake_entity_manager = utils.get_fake_entity_manager(args["data"], args["sample"], args["strategy"], args["t"], args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], None, args["anony_mode"], args)
    elif args["anony_mode"] == CLUSTERS_AND_GRAPH_ANONYMIZATION_MODE:
        fake_entity_manager = utils.get_fake_entity_manager(args["data"], args["sample"], args["strategy"], args["t"], args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], None, args["anony_mode"], args)
    else:
        raise Exception("Unsupported anony mode: {}".format(args["anony_mode"]))

    count = 0
    for cluster in clusters:
        signature = info.get_generalized_signature_info(subgraph, fake_entity_manager, cluster)

        if len(cluster) >= min_size * 2 and len(signature) >= min_signature_size * 2:
            count += 1

    return count

def calculate_num_entities_in_big_clusters(clusters, subgraph, args):
    min_size = args["k"]
    min_signature_size = args["l"]


    if args["anony_mode"] == CLUSTERS_ANONYMIZATION_MODE:
        fake_entity_manager = utils.get_fake_entity_manager(args["data"], args["sample"], args["strategy"], args["t"], args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], None, args["anony_mode"], args)
    elif args["anony_mode"] == CLUSTERS_AND_GRAPH_ANONYMIZATION_MODE:
        fake_entity_manager = utils.get_fake_entity_manager(args["data"], args["sample"], args["strategy"], args["t"], args["info_loss"], args["k"], args["w"], args["l"], args["reset_w"], args["calgo"], args["enforcer"], args["galgo"], args["anony_mode"], args)
    else:
        raise Exception("Unsupported anony mode: {}".format(args["anony_mode"]))

    count = 0
    for cluster in clusters:
        signature = info.get_generalized_signature_info(subgraph, fake_entity_manager, cluster)

        if len(cluster) >= min_size * 2 and len(signature) >= min_signature_size * 2:
            count += len(cluster)

    return count

def calculate_num_anonymized_entities(clusters, subgraph, args):
    return sum(map(lambda cluster: len(cluster), clusters))

def calculate_num_clusters(clusters, subgraph, args):
    return len(clusters)

def calculate_num_raw_entities(clusters, subgraph, args):
    return subgraph.num_entities

def calculate_num_raw_edges(clusters, subgraph, args):
    return subgraph.num_edges

clusters_metric_dict = {
    ADM_METRIC: calculate_adm,
    RADM_METRIC: calculate_radm,
    DM_METRIC: calculate_dm,
    AM_METRIC: calculate_am,
    OUT_DM_METRIC: calculate_odm,
    IN_DM_METRIC: calculate_idm,
    FAKE_ENTITIES_METRIC: calculate_num_fake_entities,
    REAL_ENTITIES_METRIC: calculate_num_real_entities,
    REAL_EDGES_METRIC: calculate_num_real_edges,
    FAKE_EDGES_METRIC: calculate_num_fake_edges,
    RAW_ENTITIES_METRIC: calculate_num_raw_entities,
    RAW_EDGES_METRIC: calculate_num_raw_edges,
    ANONYMIZED_ANONYMITY_METRIC: calculate_anonymity,
    # NUM_BIG_CLUSTERS: calculate_num_big_clusters,
    NUM_CLUSTERS: calculate_num_clusters,
    # NUM_ENTITIES_IN_BIG_CLUSTERS: calculate_num_entities_in_big_clusters,
    ANONYMIZED_ENTITIES_METRIC: calculate_num_anonymized_entities,
}

def get_all_metric_names():
    return list(clusters_metric_dict.keys())

def calculate_quality_metrics(metric_names, clusters, subgraph, args):
    quality = {}
    for metric_name in metric_names:
        fn = clusters_metric_dict[metric_name]
        quality_value = fn(clusters, subgraph, args)
        quality[metric_name] = quality_value

    return quality

