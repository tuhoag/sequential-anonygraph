import sys
import itertools
import logging
from time import time
import numpy as np
from sortedcontainers import SortedList
import copy

logger = logging.getLogger(__name__)

def add_to_group(groups, group_id, t):
    group = groups.get(group_id)
    if group is None:
        group = []
        groups[group_id] = group

    group.append(t)

def calculate_sum_differences(start_index_list, num_edges_list):
    result_sum_num_edges_list = np.zeros(shape=(len(start_index_list) - 1))

    for i in range(len(start_index_list) - 1):
        current_num_edges_list = num_edges_list[start_index_list[i]:start_index_list[i+1]]
        result_sum_num_edges_list[i] = np.sum(current_num_edges_list)
        logger.debug("i: {} - current sum: {} of {}".format(i, result_sum_num_edges_list, current_num_edges_list))

    difference = np.std(result_sum_num_edges_list)

    return difference

def find_min_dif_start_index_list(start_index_list, num_edges_list, max_index):
    min_start_index_list = None
    min_dif = sys.maxsize

    for i in range(1, len(start_index_list) - 1):
        can_increase = False
        logger.debug("current start index: {} next index: {}".format(start_index_list[i], start_index_list[i + 1]))
        if i == len(start_index_list) - 2:
            if start_index_list[i] < max_index:
                # increase last index
                can_increase = True
                logger.debug("can increase as last start index: {} < max index: {}".format(start_index_list[i], max_index))
        elif start_index_list[i] + 1 < start_index_list[i + 1]:
            # increase last index
            can_increase = True
            logger.debug("can increase as current start index {} + 1 < next start index {}".format(start_index_list[i], start_index_list[i + 1]))

        if can_increase:
            new_start_index_list = start_index_list.copy()
            new_start_index_list[i] = new_start_index_list[i] + 1
            new_dif = calculate_sum_differences(new_start_index_list, num_edges_list)
            logger.debug("new start index (dif: {}): {}".format(new_dif, new_start_index_list))

            if new_dif < min_dif:
                min_start_index_list = new_start_index_list
                min_dif = new_dif

        logger.debug("current start index list (dif: {}): {}".format(min_dif, min_start_index_list))

    return min_start_index_list, min_dif

def group_equal_size_time_instances(graph, num_subgraphs):
    time_instances = sorted(graph.time_instances)
    num_edges_list = [graph.get_num_edges(t) for t in time_instances]
    logger.debug("time insts: {}".format(time_instances))
    logger.debug("num edges: {}".format(num_edges_list))

    num_final_subgraphs = min(num_subgraphs, graph.num_time_instances)
    logger.debug("num final subgraphs: {}".format(num_final_subgraphs))

    start_index_list = list(range(0, num_final_subgraphs)) + [None]

    min_dif = calculate_sum_differences(start_index_list, num_edges_list)
    logger.info("initial start indexes (dif: {}): {}".format(min_dif, start_index_list))

    max_index = len(time_instances)
    logger.info("max index: {}".format(max_index))
    step = 0
    while(True):
        next_start_index_list, next_dif = find_min_dif_start_index_list(start_index_list, num_edges_list, max_index)

        if next_dif >= min_dif:
            break

        start_index_list = next_start_index_list
        min_dif = next_dif
        logger.info("[step: {}] current start index list (dif: {}): {}".format(step, min_dif, start_index_list))
        step += 1

    logger.info("final start_index_list (dif: {}): {}".format(min_dif, start_index_list))

    time_groups = {}
    for i in range(len(start_index_list) - 1):
        current_time_insts = time_instances[start_index_list[i]:start_index_list[i + 1]]
        time_groups[i] = current_time_insts
        logger.debug(time_groups)

    return time_groups

def calculate_assignment_cost(inst_idx2freq, assignment):
    t2freq = np.zeros(len(assignment))
    for t, inst_idxes in enumerate(assignment):
        for inst_idx in inst_idxes:
            t2freq[t] += inst_idx2freq[inst_idx]

    return np.std(t2freq)

def calculate_assignment_code(new_assignment):
    result = []
    for inst_idxes in new_assignment:
        result.append(tuple(inst_idxes))

    return tuple(result)

def group_time_instances_by_mean_edges(graph, num_subgraphs):
    time_instances = sorted(graph.time_instances)
    logger.debug("number of insts: {}".format(len(time_instances)))
    # mean_edges = graph.num_edges / num_subgraphs
    # logger.debug('mean edges: {}'.format(mean_edges))
    num_insts = len(time_instances)

    if num_subgraphs < 0:
        raise Exception("num of subgraphs: {} is less than 0".format(num_subgraphs))

    if num_insts < num_subgraphs:
        raise Exception("num of instances: {} < num of subgraphs: {}".format(num_insts, num_subgraphs))

    if num_insts == num_subgraphs:
        raise Exception("didn't implement num inst = num subgraph yet.")

    inst_idx2freq = np.zeros(num_insts)
    initial_assignment = [[] for _ in range(num_subgraphs)]

    t = 0
    for inst_idx, inst in enumerate(time_instances):
        initial_assignment[t].append(inst_idx)
        t = min(t + 1, num_subgraphs - 1)

        inst_idx2freq[inst_idx] = graph.get_num_edges(inst)


    initial_cost = calculate_assignment_cost(inst_idx2freq, initial_assignment)
    # logger.debug(inst_idx2freq)
    # logger.debug(initial_assignment)
    logger.debug(initial_cost)

    # open_set = SortedList([(initial_cost, initial_assignment)], key=lambda item: item[0])
    # seen_set = { calculate_assignment_code(initial_assignment)}

    next_solution = (sys.maxsize, None)
    best_solution = (initial_cost, initial_assignment)

    step = 0
    while(best_solution[1] is not None):
        logger.debug("[step: {}] current cost: {}".format(step, best_solution[0]))
        # logger.debug("current assignment: {}".format(best_solution[1]))
        step += 1
        # logger.debug("open: {} - seen: {}".format(len(open_set), len(seen_set)))
        # logger.debug("open: {}".format(open_set))
        # logger.debug("closed: {}".format(seen_set))

        # current_cost, current_assignment = open_set.pop(0)

        # if current_cost < best_assignment[0]:
        #     best_assignment = (current_cost, current_assignment)

        next_solution = (sys.maxsize, None)

        for t in range(1, num_subgraphs):
            inst_idxes = best_solution[1][t]
            if len(inst_idxes) > 1:
                new_assignment = copy.deepcopy(best_solution[1])
                new_assignment[t-1].append(new_assignment[t][0])
                new_assignment[t].pop(0)

                # code = calculate_assignment_code(new_assignment)

                # if code not in seen_set:
                #     seen_set.add(code)

                new_cost = calculate_assignment_cost(inst_idx2freq, new_assignment)
                if new_cost < best_solution[0] and new_cost < next_solution[0]:
                    next_solution = (new_cost, new_assignment)

        if next_solution[1] is None:
            break

        best_solution = next_solution

    logger.debug(best_solution)

    groups = {}

    for t in range(num_subgraphs):
        logger.debug(best_solution[1][t])
        for inst_idx in best_solution[1][t]:
            # logger.debug("add group: {} - inst: {}".format(t, time_instances[inst_idx]))
            add_to_group(groups, t, time_instances[inst_idx])

            # logger.debug(groups)


    logger.debug(groups)
    # raise Exception()
    return groups


def generate_graph(graph, t, group):
    logger.info("generate graph for t: {}".format(t))
    return {t: graph.generate_subgraph_from_time_instances(group)}

def get_num_edges_from_time_insts(graph, time_insts):
    return np.sum(list(map(lambda t: graph.get_num_edges(t), time_insts)))

def test_group_time_instances(groups, graph):
    num_instances = 0
    for t, group in groups.items():
        logger.debug('t: {} - num times: {} - num edges: {}'.format(t, len(group), get_num_edges_from_time_insts(graph, group)))

        num_instances += len(group)

    assert num_instances == graph.num_time_instances, "num instances: {} - origin: {}".format(
        num_instances, graph.num_time_instances
    )

