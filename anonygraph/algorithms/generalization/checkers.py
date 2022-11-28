import logging

import anonygraph.info_loss.info as info

logger = logging.getLogger(__name__)

def is_attributes_invalid(cluster, subgraph):
    union_info, users_info = info.get_generalized_attribute_info(subgraph, cluster.entity_ids)
    for _, user_info in users_info.items():
        if union_info != user_info:
            return True

    return False

def is_degree_invalid(cluster, subgraph, degree_type):
    generalized_info, entities_info = info.get_generalized_degree_info(subgraph, cluster.entity_ids, degree_type)
    return _is_degree_info_invalid(generalized_info, entities_info)

def _is_degree_info_invalid(generalized_info, entities_info):
    for relation_id in generalized_info.keys():
        if _is_relation_degree_info_invalid(relation_id, generalized_info, entities_info):
            return True

    return False

def is_relation_degree_invalid(cluster, subgraph, relation_id, degree_type):
    generalized_info, entities_info = info.get_generalized_degree_info(subgraph, cluster.entity_ids, degree_type)
    return _is_relation_degree_info_invalid(relation_id, generalized_info,
                                         entities_info)

def _is_relation_degree_info_invalid(relation_id, generalized_info, entities_info):
    generalized_degree = generalized_info.get(relation_id, 0)

    for _, entity_info in entities_info.items():
        entity_degree = entity_info.get(relation_id, 0)

        if entity_degree != generalized_degree:
            return True

    return False

def is_attributes_and_degree_invalid(cluster, subgraph):
    return is_attributes_invalid(cluster, subgraph) or is_degree_invalid(cluster, subgraph, 'out') or is_degree_invalid(cluster, subgraph, 'in')

def is_out_in_degree_invalid(cluster, subgraph):
    out_invalid = is_degree_invalid(cluster, subgraph, 'out')
    in_invalid = is_degree_invalid(cluster, subgraph, 'in')

    logger.debug("cluster: {} - out invalid: {} - in invalid: {}".format(cluster, out_invalid, in_invalid))
    return is_degree_invalid(cluster, subgraph, 'out') or is_degree_invalid(cluster, subgraph, 'in')