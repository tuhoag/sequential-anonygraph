import os
import logging
import subprocess

import anonygraph.constants as constants
import anonygraph.info_loss as ifn
import anonygraph.algorithms as algo
import anonygraph.utils.path as putils

logger = logging.getLogger(__name__)


def get_info_loss_function(info_loss_name, graph, args):
    if info_loss_name == 'am':
        info_loss_fn = ifn.AttributeInfoLoss(graph, args)
    elif info_loss_name == 'adm':
        info_loss_fn = ifn.AttributeOutInDegreeInfoLoss(graph, args)
    else:
        raise NotImplementedError(
            "Unsupported info loss: {}".format(info_loss_name)
        )

    return info_loss_fn


def get_pair_distance_of_subgraph(
    data_name, sample, strategy, time_instance, info_loss_name, args
):
    path = putils.get_pair_distance_of_subgraph_path(
        data_name, sample, strategy, time_instance, info_loss_name, args
    )
    return algo.PairDistance.from_file(path)


def get_all_time_instances(data_name, sample, strategy, args):
    subgraphs_path = putils.get_sequence_raw_subgraphs_path(
        data_name, sample, strategy, args
    )

    time_instances = []
    for path in os.listdir(subgraphs_path):
        if os.path.isdir(os.path.join(subgraphs_path, path)):
            time_instances.append(int(path))

    return time_instances


def get_history_table(
    data_name, sample, strategy, time_instance, info_loss_name, k, w, l, reset_w, calgo,
    enforcer, galgo, anony_mode, args
):
    if time_instance >= 0:
        path = putils.get_history_table_path(
            data_name, sample, strategy, time_instance, info_loss_name, k, w, l, reset_w,
            calgo, enforcer, galgo, anony_mode, args
        )
        logger.debug("loading history table from: {}".format(path))
        history = algo.HistoryTable.from_file(path)
    else:
        logger.debug("loading empty history table")
        history = algo.HistoryTable()

    return history


def get_fake_entity_manager(
    data_name, sample, strategy, time_instance, info_loss_name, k, w, l, reset_w, calgo, enforcer,
    galgo, anony_mode, args
):
    entity_index_path = putils.entity_index_path(data_name, sample)
    logger.debug("time_instance: {}".format(time_instance))
    if time_instance >= 0:
        fake_entities_path = putils.get_fake_entity_path(
            data_name, sample, strategy, time_instance, info_loss_name, k, w, l,reset_w,
            calgo, enforcer, galgo, anony_mode, args
        )
    else:
        fake_entities_path = None

    logger.debug("fake_entities_path: {}".format(fake_entities_path))
    manager = algo.FakeEntityManager.from_file(
        fake_entities_path, entity_index_path
    )
    return manager


def split_data_to_parts(data, num_parts):
    part_size, mod = divmod(len(data), num_parts)

    parts = []
    for part_idx in range(num_parts):
        start_idx = part_idx * part_size + min(mod, part_idx)
        end_idx = (part_idx + 1) * part_size + min(part_idx + 1, mod)
        part = data[start_idx:end_idx]
        parts.append(part)

    parts = list(filter(lambda item: len(item) > 0, parts))

    return parts


def merge_dictionaries(*dictionaries):
    result = dict()

    for dictionary in dictionaries:
        result.update(dictionary)

    return result