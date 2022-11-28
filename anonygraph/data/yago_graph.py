import itertools
import logging
import os

from tqdm import tqdm
import networkx as nx
from sortedcontainers import SortedSet

from .dynamic_graph import DynamicGraph

logger = logging.getLogger(__name__)


def remove_brackets(name):
    return name[1:-1]


ATTRIBUTE_RELATION_NAMES = {
    "livesIn", "worksAt", "hasOfficialLanguage", "happenedIn", "isCitizenOf",
    "playsFor", "directed", "created", "isPoliticianOf", "diedIn",
    "wroteMusicFor", "isKnownFor", "actedIn", "graduatedFrom", "worksAt",
    "hasWonPrize", "isAffiliatedTo", "wasBornIn", "isInterestedIn"
}
RELATIONSHIP_RELATION_NAMES = {
    "hasAcademicAdvisor", "isLeaderOf", "influences", "isMarriedTo", "hasChild"
}

def get_year_from_time_string(time_str):
    return int(time_str[1:-1].split("-")[0])

def extract_edges(file_path):
    all_time_edges = []
    temporal_edges = {}
    all_time_insts = SortedSet()

    supported_relations = ATTRIBUTE_RELATION_NAMES.union(RELATIONSHIP_RELATION_NAMES)

    relation2entities = {}
    entities = set()
    r2domain = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = list(map(lambda line: line.rstrip().split('\t'), lines))

        for splits in lines:
            head_str = remove_brackets(splits[0])
            relation_str = remove_brackets(splits[1])
            tail_str = remove_brackets(splits[2])

            if relation_str not in supported_relations:
                continue


            if relation_str in ATTRIBUTE_RELATION_NAMES:
                rentities = relation2entities.get(relation_str, set())
                rentities.add(head_str)
                relation2entities[relation_str] = rentities

                entities.add(head_str)

                rdomain = r2domain.get(relation_str, set())
                rdomain.add(tail_str)
                r2domain[relation_str] = rdomain
            else:
                entities.add(head_str)
                entities.add(tail_str)


            edge = (head_str, relation_str, tail_str)

            if len(splits) == 3:
                all_time_edges.append(edge)
            elif len(splits) == 5:
                edge_year = get_year_from_time_string(splits[4])
                all_time_insts.add(edge_year)
                time_relation = splits[3]

                edge_time_insts = temporal_edges.get(edge)

                if edge_time_insts is None:
                    edge_time_insts = [0, 0]
                    temporal_edges[edge] = edge_time_insts

                if time_relation == "<occursSince>":
                    edge_time_insts[0] = edge_year
                elif time_relation == "<occursUntil>":
                    edge_time_insts[1] = edge_year
                else:
                    raise Exception("Unsupported time relation: {}".format(time_relation))

                logger.debug(temporal_edges)
            else:
                logger.debug("incompleted edges: {}".format(splits))


    logger.info("num entities: {}".format(len(entities)))
    for relation, rentities in relation2entities.items():
        logger.info("rel: {} - num entities: {} - domain size: {}".format(relation, len(rentities), len(r2domain[relation])))


    # logger.info("entities: {}".format(entities))
    # logger.debug(relation2entities)
    # raise Exception()
    return all_time_edges, temporal_edges, all_time_insts

def add_edge(graph, edge, time_inst):
    head, relation, tail = edge

    if relation in ATTRIBUTE_RELATION_NAMES:
        graph.add_attribute_edge(head, relation, tail, time_inst)
    elif relation in RELATIONSHIP_RELATION_NAMES:
        graph.add_relationship_edge(head, relation, tail, time_inst)
    else:
        logger.debug("relation {} is not supported.".format(relation))

def add_all_time_edges(graph, edges, all_time_insts):
    num_edges = len(edges)
    num_time_insts = len(all_time_insts)
    num_temporal_edges = num_edges * num_time_insts
    num_added_edges = 0
    for edge, time_inst in tqdm(itertools.product(edges, all_time_insts), total=num_temporal_edges):
        add_edge(graph, edge, time_inst)
        num_added_edges += 1

    assert num_temporal_edges == num_added_edges

def get_edge_time_insts(edge_time_insts, all_time_insts):
    logger.debug("edge time: {}".format(edge_time_insts))

    first_time_inst, last_time_inst = edge_time_insts

    if first_time_inst == 0:
        first_time_inst = all_time_insts[0]

    if last_time_inst == 0:
        last_time_inst = all_time_insts[-1]

    first_time_index = all_time_insts.index(first_time_inst)
    last_time_index = all_time_insts.index(last_time_inst)

    logger.debug("found {} at {}".format(first_time_inst, first_time_index))
    logger.debug("found {} at {}".format(last_time_inst, last_time_index))
    logger.debug(all_time_insts)

    if last_time_index == len(all_time_insts):
        edge_time_insts = all_time_insts[first_time_index:last_time_index]
    else:
        edge_time_insts = all_time_insts[first_time_index:]

    logger.debug("final edge time insts: {}".format(edge_time_insts))

    # if first_time_inst == 0 or last_time_inst == 0:
    #     raise Exception()

    return edge_time_insts


def add_temporal_edges(graph, edges, all_time_insts):
    for edge, time_insts in tqdm(iterable=edges.items(), total=len(edges)):
        edge_time_insts = get_edge_time_insts(time_insts, all_time_insts)

        for time_inst in edge_time_insts:
            add_edge(graph, edge, time_inst)

def load_users_relationship(graph, file_path):
    logger.info("extracting edges")
    all_time_edges, temporal_edges, all_time_insts = extract_edges(file_path)
    logger.info("extracted {} edges, {} temporal edges.".format(len(all_time_edges), len(temporal_edges)))

    logger.info("adding all time edges")
    logger.info(graph)
    add_all_time_edges(graph, all_time_edges, all_time_insts)
    logger.info(graph)

    logger.info("adding temporal edges")
    add_temporal_edges(graph, temporal_edges, all_time_insts)

    # raise Exception()


class YagoGraph(DynamicGraph):
    def __init__(self):
        super().__init__()

    @staticmethod
    def from_raw_file(data_dir, args):
        # load attributes
        # load relationships
        graph = YagoGraph()
        user_relationship_path = os.path.join(data_dir, 'yago15k_train.txt')
        load_users_relationship(graph, user_relationship_path)

        return graph
