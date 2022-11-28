def calculate_num_entities(subgraph, pre_subgraph, args):
    return subgraph.num_entities

def calculate_num_edges(subgraph, pre_subgraph, args):
    return subgraph.num_edges

def calculate_new_entities(subgraph, pre_subgraph, args):
    count = 0
    for entity_id in subgraph.entity_ids:
        if not pre_subgraph.is_entity_id(entity_id):
            count += 1

    return count

def calculate_new_edges(subgraph, pre_subgraph, args):
    count = 0
    for entity1_id, relation_id, entity2_id in subgraph.get_edges_iter():
        if not pre_subgraph.is_edge_existed(entity1_id, relation_id, entity2_id):
            count += 1

    return count

def calculate_removed_entities(subgraph, pre_subgraph, args):
    count = 0

    for entity_id in pre_subgraph.entity_ids:
        if not subgraph.is_entity_id(entity_id):
            count += 1

    return count

def calculate_old_entities(subgraph, pre_subgraph, args):
    count = 0

    for entity_id in pre_subgraph.entity_ids:
        if subgraph.is_entity_id(entity_id):
            count += 1

    return count

def calculate_removed_edges(subgraph, pre_subgraph, args):
    count = 0
    for entity1_id, relation_id, entity2_id in subgraph.get_edges_iter():
        if subgraph.is_edge_existed(entity1_id, relation_id, entity2_id) and not pre_subgraph.is_edge_existed(entity1_id, relation_id, entity2_id):
            count += 1

    return count

raw_subgraph_metric_dict = {
    "num_entities": calculate_num_entities,
    "num_edges": calculate_num_edges,
    "num_new_entities": calculate_new_entities,
    "num_old_entities": calculate_old_entities,
    "num_removed_entities": calculate_removed_entities,
    "num_new_edges": calculate_new_edges,
    "num_removed_edges": calculate_removed_edges,
    # "ratio_changed_entities":
}


def get_all_metric_names():
    return list(raw_subgraph_metric_dict.keys())


def calculate_quality_metrics(metric_names, subgraph, pre_subgraph, args):
    quality = {}
    for metric_name in metric_names:
        fn = raw_subgraph_metric_dict[metric_name]
        quality_value = fn(subgraph, pre_subgraph, args)
        quality[metric_name] = quality_value

    return quality

