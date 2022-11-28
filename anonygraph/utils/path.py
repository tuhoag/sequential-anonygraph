import os
import logging

from networkx.algorithms.traversal.depth_first_search import dfs_edges

from anonygraph import settings
from anonygraph.constants import *
from anonygraph.utils import data as dutils

logger = logging.getLogger(__name__)


def get_raw_data_path(data_name):
    return os.path.join(settings.RAW_DATA_PATH, data_name)


def get_raw_graph_dir_name(data_name, sample):
    return "{}_{}".format(data_name, sample)


def get_output_path(data_name, sample):
    return os.path.join(
        settings.OUTPUT_DATA_PATH, get_raw_graph_dir_name(data_name, sample)
    )


def get_raw_graph_path(data_name, sample):
    output_dir = os.path.join(get_output_path(data_name, sample), "raw")
    return output_dir

def get_domain_data_path(data_name, sample):
    raw_graph_path = get_raw_graph_path(data_name, sample)
    logger.debug("raw_graph_path: {}".format(raw_graph_path))

    return os.path.join(raw_graph_path, "domains.txt")

def get_attr_name2id_path(data_name, sample):
    raw_graph_path = get_raw_graph_path(data_name, sample)

    return os.path.join(raw_graph_path, "attrs.idx")

def get_strategy_name(strategy, args):
    sensitive_attr = dutils.get_sensitive_attribute_name(args["data"], args["sattr"])

    if strategy in [RAW_STRATEGY, STATIC_STRATEGY]:
        logger.debug(args)
        strategy_name = strategy
    elif strategy == PERIOD_GEN_STRATEGY:
        strategy_name = "{}_{}_{}".format(
            strategy, args["period"], args["unit"]
        )
    elif strategy in [
        EQUAL_ADDITION_SIZE_STRATEGY,
        EQUAL_RAW_SIZE_STRATEGY,
        RAW_ADDITION_STRATEGY,
        MEAN_ADDITION_EDGES_STRATEGY,
        MEAN_EDGES_STRATEGY,
    ]:
        # logger.debug(args)
        strategy_name = "{}_{}".format(strategy, args["n_sg"])
    else:
        raise NotImplementedError("Unsupported strategy: {}".format(strategy))

    strategy_name = "{}_{}".format(sensitive_attr, strategy_name)
    return strategy_name


def get_sequence_subgraphs_path(data_name, sample, strategy, args):
    output_path = get_output_path(data_name, sample)
    strategy_name = get_strategy_name(strategy, args)

    sequence_path = os.path.join(output_path, strategy_name)

    return sequence_path

def get_anonymized_subgraph_entity2sensitive_vals_path(data_name, sample, strategy, time_instance, info_loss_name, k, w, l, reset_w, calgo_name, enforcer_name, galgo_name, anony_mode, args):
    return os.path.join(get_anonymized_subgraph_path(data_name, sample, strategy, time_instance, info_loss_name, k, w, l, reset_w, calgo_name, enforcer_name, galgo_name, anony_mode, args), "sensitive.vals")

def get_raw_subgraph_entity2sensitive_vals_path(data_name, sample, strategy, time_instance, args):
    return os.path.join(get_raw_subgraph_path(data_name, sample, strategy, time_instance, args), "sensitive.vals")


def get_sequence_raw_subgraphs_path(data_name, sample, strategy, args):
    sequence_path = get_sequence_subgraphs_path(
        data_name, sample, strategy, args
    )
    sequence_raw_path = os.path.join(sequence_path, "raw")

    return sequence_raw_path


def get_raw_subgraph_path(data_name, sample, strategy, time_instance, args):
    sequence_raw_path = get_sequence_raw_subgraphs_path(
        data_name, sample, strategy, args
    )
    sequence_raw_path = os.path.join(
        sequence_raw_path, "{:03d}".format(time_instance)
    )

    return sequence_raw_path

def get_extra_sval_edges_path(data_name, sample, strategy, args):
    sequence_path = get_sequence_raw_subgraphs_path(
        data_name, sample, strategy, args
    )

    sattr_name = dutils.get_sensitive_attribute_name(data_name, args["sattr"])

    return os.path.join(sequence_path, "extra_sattr_edges.txt".format(sattr_name))



def get_time_group_path(data_name, sample, strategy, args):
    sequence_path = get_sequence_raw_subgraphs_path(
        data_name, sample, strategy, args
    )
    return os.path.join(sequence_path, "time_groups.txt")

def get_pair_distance_path(data_name, sample, strategy, args):
    sequence_path = get_sequence_subgraphs_path(
        data_name, sample, strategy, args
    )
    pair_dist_path = os.path.join(sequence_path, "pair_dist")
    return pair_dist_path


def get_info_loss_full_string(info_loss_name, args):
    if info_loss_name == "adm":
        name = "{}#{:04.2f}#{:04.2f}".format(
            info_loss_name, args["alpha_adm"], args["alpha_dm"])
    else:
        raise NotImplementedError(
            "Unsupported info loss metric: {}".format(info_loss_name)
        )

    return name


def get_pair_distance_of_subgraph_path(
    data_name, sample, strategy, time_instance, info_loss_name, args
):
    inst_subgraph_path = get_pair_distance_path(
        data_name, sample, strategy, args
    )
    info_loss_str = get_info_loss_full_string(info_loss_name, args)
    path = os.path.join(
        inst_subgraph_path,
        "{ifn}_{t:02d}.txt".format(t=time_instance, ifn=info_loss_str)
    )
    return path

def get_anonymization_outputs_path(data_name, sample, strategy, anony_mode, args):
    sequence_path = get_sequence_subgraphs_path(
        data_name, sample, strategy, args
    )

    path = os.path.join(sequence_path, anony_mode)
    return path

def get_history_tables_path(data_name, sample, strategy, anony_mode, args):
    anony_outputs_path = get_anonymization_outputs_path(
        data_name, sample, strategy, anony_mode, args
    )
    history_path = os.path.join(anony_outputs_path, "histories")
    return history_path

def get_constraint_str(k, w, l, reset_w):
    return "{:02d}#{:02d}#{:02d}#{:02d}".format(k, w, l, reset_w)


def get_history_table_path(
    data_name, sample, strategy, time_instance, info_loss_name, k, w, l, reset_w, calgo, enforcer,
    galgo, anony_mode, args
):
    histories_path = get_history_tables_path(data_name, sample, strategy, anony_mode, args)

    if anony_mode == CLUSTERS_AND_GRAPH_ANONYMIZATION_MODE:
        file_name = get_anonymized_subgraph_name(
            time_instance, info_loss_name, k, w, l, reset_w, calgo, enforcer, galgo, args
        )
    elif anony_mode == CLUSTERS_ANONYMIZATION_MODE:
        file_name = get_clusters_file_name(time_instance, info_loss_name, k, w, l, reset_w, calgo, enforcer, args)
    else:
        raise Exception("Unsupported anony mode: {}".format(anony_mode))

    history_path = os.path.join(
        histories_path, "{}.json".format(file_name)
    )
    return history_path

def get_all_clusters_path(data_name, sample, strategy, anony_mode, args):
    anony_outputs_path = get_anonymization_outputs_path(
        data_name, sample, strategy, anony_mode, args
    )

    history_path = os.path.join(anony_outputs_path, "clusters")
    return history_path


def get_clustering_algorithm_str(calgo, args):
    if calgo == "random":
        return "random"
    elif calgo in [KMEDOIDS_CLUSTERING, CUSTOM_CLUSTERING, HDBSCAN_CLUSTERING]:
        return "{}".format(calgo)
    else:
        raise NotImplementedError(
            "Unsupported clustering algo: {}".format(calgo)
        )

def get_enforcer_str(enforcer_name, args):
    # logger.debug(enforcer_name)
    # logger.debug(args)
    if enforcer_name == INVALID_REMOVAL_ENFORCER:
        return enforcer_name
    elif enforcer_name == MERGE_SPLIT_ASSIGNMENT_ENFORCER:
        return "{}#{:03.2f}".format(enforcer_name, args["max_dist"])
    elif enforcer_name == SPLIT_OVERLAP_ASSIGNMENT_ENFORCER:
        return "{}#{:03.2f}".format(enforcer_name, args["max_dist"])
    elif enforcer_name == GREEDY_SPLIT_ENFORCER:
        return "{}#{:03.2f}".format(enforcer_name, args["max_dist"])
    else:
        raise Exception("Do not support '{}' enforcer.".format(enforcer_name))


def get_clusters_file_name(time_instance, info_loss_name, k, w, l, reset_w, calgo_name, enforcer_name, args):
    info_loss_str = get_info_loss_full_string(info_loss_name, args)
    constraint_str = get_constraint_str(k, w, l, reset_w)
    algo_str = get_clustering_algorithm_str(calgo_name, args)
    enforcer_str = get_enforcer_str(enforcer_name, args)
    return "{ifn}_{constraint}_{calgo}_{enforcer}_{t:03d}".format(
        ifn=info_loss_str,
        constraint=constraint_str,
        calgo=algo_str,
        enforcer=enforcer_str,
        t=time_instance
    )


def get_clusters_path(
    data_name, sample, strategy, time_instance, info_loss_name, k, w, l, reset_w, calgo_name, enforcer_name, anony_mode,
    args
):
    clusters_path = get_all_clusters_path(data_name, sample, strategy, anony_mode, args)
    clusters_file_name = get_clusters_file_name(
        time_instance, info_loss_name, k, w, l, reset_w, calgo_name, enforcer_name, args
    )

    cluster_path = os.path.join(
        clusters_path, "{}.txt".format(clusters_file_name)
    )

    return cluster_path


def get_fake_entities_path(data_name, sample, strategy, anony_mode, args):
    anony_outputs_path = get_anonymization_outputs_path(data_name, sample, strategy, anony_mode, args)
    return os.path.join(anony_outputs_path, "fake_entities")


def entity_index_path(data_name, sample):
    raw_graph_path = get_raw_graph_path(data_name, sample)
    return os.path.join(raw_graph_path, "entities.idx")


def get_fake_entity_path(
    data_name, sample, strategy, time_instance, info_loss_name, k, w, l, reset_w, calgo_name, enforcer_name,
    galgo_name, anony_mode, args
):
    fake_entities_path = get_fake_entities_path(
        data_name, sample, strategy, anony_mode, args
    )

    if anony_mode == CLUSTERS_AND_GRAPH_ANONYMIZATION_MODE:
        # name = get_anonymized_subgraph_name(
        #     time_instance, info_loss_name, k, w, l, reset_w, calgo_name, enforcer_name, galgo_name, args
        # )
        name = get_clusters_file_name(
            time_instance, info_loss_name, k, w, l, reset_w, calgo_name, enforcer_name, args
        )
    elif anony_mode == CLUSTERS_ANONYMIZATION_MODE:
        name = get_clusters_file_name(
            time_instance, info_loss_name, k, w, l, reset_w, calgo_name, enforcer_name, args
        )
    else:
        raise Exception("Unsupported anony mode: {}".format(anony_mode))

    logger.debug("name: {}".format(name))
    return os.path.join(fake_entities_path, "{}.idx".format(name))


def get_all_anonymized_subgraphs_dir(data_name, sample, strategy, anony_mode, args):
    anony_outputs_path = get_anonymization_outputs_path(data_name, sample, strategy, anony_mode, args)
    path = os.path.join(anony_outputs_path, "graphs")
    return path


def get_generalization_string(galgo, args):
    if galgo in [ADD_REMOVE_EDGES_GEN, ADD_REMOVE_EDGES2_GEN]:
        return "{}".format(galgo)
    else:
        raise NotImplementedError(
            "Unsupported generalization algo: {}".format(galgo)
        )


def get_anonymized_subgraph_name(
    time_instance, info_loss_name, k, w, l, reset_w, calgo, enforcer, galgo, args
):
    cluster_name = get_clusters_file_name(
        time_instance, info_loss_name, k, w, l, reset_w, calgo, enforcer, args
    )
    gen_name = get_generalization_string(galgo, args)
    anonymized_subgraph_name = "{}_{}".format(cluster_name, gen_name)

    return anonymized_subgraph_name


def get_anonymized_subgraph_path(
    data_name, sample, strategy, time_instance, info_loss_name, k, w, l, reset_w, calgo, enforcer,
    galgo, anony_mode, args
):
    all_subgraph_dir = get_all_anonymized_subgraphs_dir(
        data_name, sample, strategy, anony_mode, args
    )

    anonymized_subgraph_name = get_anonymized_subgraph_name(
        time_instance, info_loss_name, k, w, l, reset_w, calgo, enforcer, galgo, args
    )

    return os.path.join(all_subgraph_dir, anonymized_subgraph_name)


def get_dynamic_graph_exp_data_path(data_name, sample):
    data_name = get_raw_graph_dir_name(data_name, sample)
    return os.path.join(
        settings.EXP_DATA_PATH, "dynamic_graph_info",
        "{}.csv".format(data_name)
    )

def get_performance_stats_exp_data_path(data_name, sample, strategy, anony_mode, args):
    data_name = get_raw_graph_dir_name(data_name, sample)
    strategy_name = get_strategy_name(strategy, args)

    return os.path.join(
        settings.EXP_DATA_PATH, "performance",
        "{}#{}#{}.csv".format(data_name, strategy_name, anony_mode)
    )

def get_comparision_exp_data_path(data_name, sample, strategy, anony_mode, args):
    data_name = get_raw_graph_dir_name(data_name, sample)
    strategy_name = get_strategy_name(strategy, args)

    return os.path.join(
        settings.EXP_DATA_PATH, "comparison",
        "{}#{}#{}.csv".format(data_name, strategy_name, anony_mode)
    )

def get_tuning_clusters_exp_data_path(data_name, sample, strategy, anony_mode, args):
    data_name = get_raw_graph_dir_name(data_name, sample)
    strategy_name = get_strategy_name(strategy, args)

    return os.path.join(
        settings.EXP_DATA_PATH, "tuning_clusters",
        "{}#{}#{}.csv".format(data_name, strategy_name, anony_mode)
    )


def get_tuning_graphs_exp_data_path(data_name, sample, strategy, args):
    data_name = get_raw_graph_dir_name(data_name, sample)
    strategy_name = get_strategy_name(strategy, args)

    return os.path.join(
        settings.EXP_DATA_PATH, "tuning_graphs",
        "{}#{}.csv".format(data_name, strategy_name)
    )


def get_raw_snapshots_exp_data_path(data_name, sample, args):
    data_name = get_raw_graph_dir_name(data_name, sample)
    # strategy_name = get_strategy_name(strategy, args)
    return os.path.join(
        settings.EXP_DATA_PATH, "raw_snapshots", "{}.csv".format(data_name)
    )

def get_all_performance_stats_path(data_name, sample, strategy, anony_mode, args):
    anony_outputs_path = get_anonymization_outputs_path(
        data_name, sample, strategy, anony_mode, args
    )

    return os.path.join(anony_outputs_path, "stats")

def get_performance_stats_file_name(info_loss_name, k, w, l, reset_w, calgo, galgo, args):
    info_loss_str = get_info_loss_full_string(info_loss_name, args)
    constraint_str = get_constraint_str(k, w, l, reset_w)

    algo_str = get_clustering_algorithm_str(calgo, args)
    gen_name = get_generalization_string(galgo, args)

    return "{ifn}_{constraint}_{calgo}_{galgo}".format(
        ifn=info_loss_str,
        constraint=constraint_str,
        calgo=algo_str,
        galgo=galgo,
    )

def get_performance_stats_data_path(
    data_name, sample, strategy, info_loss_name, k, w, l, reset_w, calgo, enforcer, galgo, anony_mode, time_instance, args
):
    stats_path = get_all_performance_stats_path(data_name, sample, strategy, anony_mode, args)
    stats_file_name = get_anonymized_subgraph_name(
        time_instance, info_loss_name, k, w, l, reset_w, calgo, enforcer, galgo, args
    )

    stats_path = os.path.join(
        stats_path, "{}.csv".format(stats_file_name)
    )

    return stats_path

def extract_info_from_data_name(data_name):
    splits = data_name.split("_")
    return {"data": splits[0], "sample": int(splits[1])}


def extract_info_from_strategy_name(strategy_name):
    splits = strategy_name.split("_")
    sattr = splits[0]
    strategy_name = splits[1]

    info = {
        "sattr": sattr,
        "strategy": strategy_name,
    }
    if strategy_name == PERIOD_GEN_STRATEGY:
        info.update({
            "period": int(splits[2]),
            "unit": splits[3],
        })
    elif strategy_name in [
        EQUAL_ADDITION_SIZE_STRATEGY,
        EQUAL_RAW_SIZE_STRATEGY,
        RAW_ADDITION_STRATEGY,
        MEAN_ADDITION_EDGES_STRATEGY,
        MEAN_EDGES_STRATEGY,
    ]:
        info.update({"n_sg": int(splits[2])})
    elif strategy_name in [RAW_STRATEGY, STATIC_STRATEGY]:
        pass
    else:
        raise NotImplementedError("Unsupported strategy: {}".format(strategy_name))
    return info


def extract_info_from_info_loss_str(info_loss_name):
    splits = info_loss_name.split("#")
    return {
        "info_loss": splits[0],
        "alpha_adm": float(splits[1]),
        "alpha_dm": float(splits[2])
    }


def extract_info_from_constraint_str(constraint_str):
    splits = constraint_str.split("#")
    return {
        "k": int(splits[0]),
        "w": int(splits[1]),
        "l": int(splits[2]),
        "reset_w": int(splits[3]),
    }


def extract_info_from_calgo_str(calgo_str):
    splits = calgo_str.split("#")
    return {
        "calgo": splits[0],
        # "max_dist": float(splits[1]),
    }

def extract_info_from_enforcer_str(enforcer_str):
    splits = enforcer_str.split("#")

    info = {
        "enforcer_str": enforcer_str,
        "enforcer": splits[0],
        "enforcer_name": splits[0],
        "max_dist": 1,
    }

    if splits[0] == GREEDY_SPLIT_ENFORCER:
        info.update({
            "max_dist": float(splits[1]),
        })

    return info

def extract_info_from_clusters_name(cluster_name):
    splits = cluster_name.split("_")
    logger.debug("splits of cluster name {}: {}".format(cluster_name, splits))

    info_loss_str = splits[0]
    constraint_str = splits[1]
    calgo_str = splits[2]
    enforcer_str = splits[3]
    t = int(splits[4])

    info = extract_info_from_info_loss_str(info_loss_str)
    info.update(extract_info_from_constraint_str(constraint_str))
    info.update(extract_info_from_calgo_str(calgo_str))
    info.update(extract_info_from_enforcer_str(enforcer_str))
    info["t"] = t
    info["algo"] = DYNAMIC_KG_ANONYMIZATION_ALGORITHM_NAME

    # logger.debug("cluster_name: {}".format(cluster_name))
    # logger.debug("info: {}".format(info))
    # raise Exception()
    return info

def extract_info_from_anony_mode_name(anony_mode_name):
    return {"anony_mode": anony_mode_name}

def extract_info_from_clusters_path(clusters_path):
    splits = clusters_path.split(".txt")[0].split("/")
    # logger.debug(splits)
    # raise Exception(splits)

    info = extract_info_from_data_name(splits[2])
    info.update(extract_info_from_strategy_name(splits[3]))
    info.update(extract_info_from_anony_mode_name(splits[4]))
    info.update(extract_info_from_clusters_name(splits[6]))
    return info

def extract_info_from_stats_name(stats_name):
    splits = stats_name.split("_")
    # raise Exception(splits)

    info_loss_name = splits[0]
    constraint_name = splits[1]
    calgo_name = splits[2]
    galgo_name = splits[3]

    info = extract_info_from_info_loss_str(info_loss_name)
    info.update(extract_info_from_constraint_str(constraint_name))
    info.update(extract_info_from_calgo_str(calgo_name))
    info.update(extract_info_from_generalization_name(galgo_name))

    return info

def extract_info_from_stats_path(stats_path):
    splits = stats_path.split(".csv")[0].split("/")
    # logger.debug(splits)
    # raise Exception(splits)

    info = extract_info_from_data_name(splits[2])
    info.update(extract_info_from_strategy_name(splits[3]))
    info.update(extract_info_from_anony_mode_name(splits[4]))
    info.update(extract_info_from_anonymized_graph_name(splits[6]))
    return info


def extract_info_from_generalization_name(generalization_name):
    splits = generalization_name.split("_")

    if splits[0] in [ADD_REMOVE_EDGES_GEN, ADD_REMOVE_EDGES2_GEN]:
        return {
            "galgo": splits[0],
        }
    else:
        raise NotImplementedError(
            "Unsupported generalization algo: {}".format(splits[0])
        )


def extract_info_from_anonymized_graph_name(graph_name):
    splits = graph_name.split("_")
    clusters_name = "_".join(splits[0:-1])
    info = extract_info_from_clusters_name(clusters_name)
    info.update(extract_info_from_generalization_name(splits[-1]))

    return info


def extract_info_from_anonymized_subgraph_path(graph_path):
    splits = graph_path.split("/")
    # raise Exception(splits)
    info = extract_info_from_data_name(splits[2])
    info.update(extract_info_from_strategy_name(splits[3]))
    info.update(extract_info_from_anony_mode_name(splits[4]))
    info.update(extract_info_from_anonymized_graph_name(splits[6]))
    return info


def extract_info_from_raw_subgraph_path(subgraph_path):
    splits = subgraph_path.split(os.sep)
    logger.debug("path: {} - splits: {}".format(subgraph_path, splits))

    t = int(splits[5])

    info = extract_info_from_data_name(splits[2])
    info.update(extract_info_from_strategy_name(splits[3]))
    info.update({"t": t})

    logger.debug(splits)
    return info

def get_training_result_file_dir():
    path = os.path.join("exp_data","training")

    return path

def get_raw_graph_training_result_file_path(data_name, strategy_name, d, t, args):
    data_str = get_raw_graph_dir_name(data_name, args["sample"])
    strategy_str = get_strategy_name(strategy_name, args)

    path = os.path.join("exp_data","training","{}_{}_{}".format(data_str, strategy_str,d), "raw", "{}.json".format(str(t)))
    return path

def get_anony_graph_training_result_file_path(data_name, strategy_name, d, k,l,reset_w,calgo,enforcer_str, t, args):
    key = "{}_{}_{}_{}_{}".format(k,l,reset_w,calgo,enforcer_str)
    data_str = get_raw_graph_dir_name(data_name, args["sample"])
    strategy_str = get_strategy_name(strategy_name, args)

    path = os.path.join("exp_data","training","{}_{}_{}".format(data_str, strategy_str,d), key, "{}.json".format(str(t)))

    return path

def get_agg_training_data_path(data_name, strategy_name, args):
    strategy_str = get_strategy_name(strategy_name, args)
    path = os.path.join("exp_data", "training", "{}_{}.csv".format(data_name, strategy_str))
    return path