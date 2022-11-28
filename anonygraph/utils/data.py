import os
import logging

from anonygraph.data import DynamicGraph
import anonygraph.data as data
import anonygraph.utils.path as putils

logger = logging.getLogger(__name__)

def load_graph_from_raw_data(data_name, sample, args):
    raw_dir = putils.get_raw_data_path(data_name)
    data_fn_dict = {
        "email-temp": data.EmailTempGraph,
        "dummy": data.DummyDynamicGraph,
        "yago15": data.YagoGraph,
        "icews14": data.ICEWS14Graph,
        "dblp": data.DBLPGraph,
        "freebase": data.FreebaseGraph,
        "gplus": data.GplusGraph,
        "email": data.EmailGraph,
    }

    data_fn = data_fn_dict.get(data_name)
    if data_fn is not None:
        graph = data_fn.from_raw_file(raw_dir, args)
    else:
        raise NotImplementedError("Unsupported graph: {}".format(data_name))

    return graph

def load_dynamic_graph_from_output_file(data_name, sample):
    output_path = putils.get_raw_graph_path(data_name, sample)
    graph = DynamicGraph.from_raw_graph_output(output_path)

    src_unit = get_data_time_unit(data_name)
    graph.time_unit = src_unit

    return graph

def get_data_time_unit(data_name):
    data2unit = {
        "email-temp": "second",
        "dummy": "week",
        "yago15": "year",
        "icews14": "year",
        "dblp": "day",
        "gplus": "day",
        "freebase": "day",
        "email": "day",
    }

    return data2unit[data_name]

def load_subgraph_metadata(data_name, sample):
    raw_dynamic_graph_path = putils.get_raw_graph_path(data_name, sample)
    node2id, relation2id, _, _, attribute_relation_ids, relationship_relation_ids = data.dynamic_graph.read_index_data(
        raw_dynamic_graph_path)
    attribute_domains = data.dynamic_graph.read_domains_data(
        raw_dynamic_graph_path)

    return node2id, relation2id, attribute_relation_ids, relationship_relation_ids, attribute_domains

def load_edges_iter_from_path(path):
    rel_info_path = os.path.join(path, "rels.edges")
    attr_info_path = os.path.join(path, "attrs.edges")

    attribute_edges_iter = data.subgraph.get_edges_iter(attr_info_path)
    relationship_edges_iter = data.subgraph.get_edges_iter(rel_info_path)

    return attribute_edges_iter, relationship_edges_iter

def load_sensitive_vals_from_path(path):
    # sensitive_vals_path = os.path.join(path, "sensitive.vals")

    with open(path, "r") as f:
        sensitive_attr_id = int(f.readline())
        entity2sensitive_vals = {}

        for line in f:
            entity_id, val_id = list(map(int, line.split(",")))
            sensitive_vals = entity2sensitive_vals.get(entity_id)

            if sensitive_vals is None:
                sensitive_vals = set()
                entity2sensitive_vals[entity_id] = sensitive_vals

            sensitive_vals.add(val_id)

        return sensitive_attr_id, entity2sensitive_vals

def load_raw_subgraph(data_name, sample, strategy, time_instance, args):
    node2id, relation2id, attribute_relation_ids, relationship_relation_ids, attribute_domains = load_subgraph_metadata(data_name, sample)

    raw_subgraph_path = putils.get_raw_subgraph_path(
        data_name, sample, strategy, time_instance, args)
    attribute_edges_iter, relationship_edges_iter = load_edges_iter_from_path(raw_subgraph_path)

    sensitive_attr_id, entity2sensitive_vals = load_sensitive_vals(data_name, sample, strategy, time_instance, args)

    return data.SubGraph.from_index_and_edges_data(node2id, relation2id, relationship_relation_ids, attribute_relation_ids, attribute_domains, attribute_edges_iter, relationship_edges_iter, sensitive_attr_id, entity2sensitive_vals)

def get_anonymized_subgraph(data_name, sample, strategy, time_instance, info_loss_name, k, w, l, reset_w, calgo, enforcer_name, galgo, anony_mode, args):
    node2id, relation2id, attribute_relation_ids, relationship_relation_ids, attribute_domains = load_subgraph_metadata(data_name, sample)

    path = putils.get_anonymized_subgraph_path(data_name, sample, strategy, time_instance, info_loss_name, k, w, l, reset_w, calgo, enforcer_name, galgo, anony_mode, args)
    attribute_edges_iter, relationship_edges_iter = load_edges_iter_from_path(path)

    sensitive_attr_id, entity2svals = load_anonymized_sensitive_vals(data_name, sample, strategy, time_instance, info_loss_name, k, w, l, reset_w, calgo, enforcer_name, galgo, anony_mode, args)

    return data.SubGraph.from_index_and_edges_data(node2id, relation2id, relationship_relation_ids, attribute_relation_ids, attribute_domains, attribute_edges_iter, relationship_edges_iter, sensitive_attr_id, entity2svals)

def get_raw_entity_indexes(data_name, sample):
    output_path = putils.get_raw_graph_path(data_name, sample)
    entity_index_path = os.path.join(output_path, "entities.idx")

    result = {}
    with open(entity_index_path) as f:
        for line in f:
            # logger.debug(line)
            splits = line.strip().split(",")
            # logger.debug(splits)

            entity_name = ",".join(splits[0:-1])
            entity_id = splits[-1]

            result[entity_name] = int(entity_id)

    return result

def load_anonymized_sensitive_vals(data_name, sample, strategy, time_instance, info_loss_name, k, w, l, reset_w, calgo_name, enforcer_name, galgo_name, anony_mode, args):
    path = putils.get_anonymized_subgraph_entity2sensitive_vals_path(data_name, sample, strategy, time_instance, info_loss_name, k, w, l, reset_w, calgo_name, enforcer_name, galgo_name, anony_mode, args)
    return load_sensitive_vals_from_path(path)

def load_sensitive_vals(data_name, sample, strategy, time_instance, args):
    path = putils.get_raw_subgraph_entity2sensitive_vals_path(data_name, sample, strategy, time_instance, args)
    return load_sensitive_vals_from_path(path)

def load_all_sensitive_vals(data_name, sample, sattr_name, args):
    sattr_name = get_sensitive_attribute_name(data_name, sattr_name)
    domain_data = load_domain_data(data_name, sample)
    attr_name2id = load_index_file(putils.get_attr_name2id_path(data_name, sample))
    sattr_id = attr_name2id[sattr_name + "_attr"]

    all_sval_ids = domain_data[sattr_id]
    logger.debug("all_sval_ids: {}".format(all_sval_ids))

    return all_sval_ids


def load_index_file(path):
    index_data = {}
    with open(path, "r") as f:
        for line in f:
            name, index = line.strip().split(",")

            index_data[name] = int(index)

    logger.debug("index_data: {}".format(index_data))
    return index_data


def load_domain_data(data_name, sample):
    domain_path = putils.get_domain_data_path(data_name, sample)
    logger.debug("domain_path: {}".format(domain_path))

    domain_data = {}
    with open(domain_path, "r") as f:
        for line in f:
            splits = line.strip().split(":")
            sattr_id = int(splits[0])
            sval_ids = set(map(int, splits[1].split(",")))

            # logger.debug(splits)
            logger.debug("sattr_id: {} - sval_ids: {}".format(sattr_id, sval_ids))

            domain_data[sattr_id] = sval_ids

    return domain_data

def get_sensitive_attribute_name(data_name, sattr_name):
    if sattr_name is not None:
        return sattr_name

    dname2sattr = {
        "email-temp": "dept",
        "yago15": "isCitizenOf",
        "dummy": "class",
        "email": "dept",
        "freebase": "location",
    }

    sattr_name = dname2sattr.get(data_name)

    if sattr_name is None:
        raise Exception("cannot find sensitive attr for data name: {}".format(data_name))

    return dname2sattr[data_name]

def read_time_groups_from_file(path):
    time_groups = {}

    with open(path, "r") as f:
        for line_idx, line in enumerate(f):
            splits = line.strip().split(",")
            time_groups[line_idx] = [int(t_str) for t_str in splits]

    return time_groups

def read_time_groups(data_name, sample, strategy_name, args):
    path = putils.get_time_group_path(
        data_name, sample, strategy_name, args
    )

    return read_time_groups_from_file(path)