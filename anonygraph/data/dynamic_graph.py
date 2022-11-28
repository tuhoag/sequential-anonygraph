from time import time
import os
from abc import ABC, abstractmethod
import logging
import networkx as nx
import numpy as np

from anonygraph.constants import *
from .subgraph import SubGraph

logger = logging.getLogger(__file__)



class DynamicGraph(ABC):
    def __init__(self):
        self.__graphs = {}
        self.__node2id = {}
        self.__relation2id = {}
        self.__relationship_relation_ids = set()
        self.__attribute_relation_ids = set()
        self.__entity_ids = set()
        self.__value_ids = set()
        self.__attribute_domains = {}
        self.__time_unit = None
        self.__num_relationship_edges = 0
        self.__num_attribute_edges = 0

    @property
    def entity_ids(self):
        return self.__entity_ids

    @property
    def time_unit(self):
        return self.__time_unit

    @time_unit.setter
    def time_unit(self, unit):
        self.__time_unit = unit

    @property
    def time_instances(self):
        return list(self.__graphs.keys())

    @property
    def num_nodes(self):
        return len(self.__node2id)

    @property
    def num_entities(self):
        return len(self.__entity_ids)

    @property
    def num_values(self):
        return len(self.__value_ids)

    @property
    def num_relations(self):
        return len(self.__relation2id)

    @property
    def attribute_relation_ids(self):
        return self.__attribute_relation_ids
    @property
    def num_attribute_relations(self):
        return len(self.__attribute_relation_ids)

    @property
    def num_relationship_relations(self):
        return len(self.__relationship_relation_ids)

    @property
    def num_time_instances(self):
        return len(self.__graphs)

    @property
    def num_edges(self):
        num_edges = 0

        for _, graph in self.__graphs.items():
            num_edges += graph.number_of_edges()

        return num_edges

    @property
    def num_attribute_edges(self):
        return self.__num_attribute_edges

    @property
    def num_relationship_edges(self):
        return self.__num_relationship_edges

    def __str__(self):
        return ("number of nodes: {:,d} (entities: {:,d} - values: {:,d})\n"
                "number of relations: {:,d} (relationships: {:,d} - attributes: {:,d})\n"
                "number of edges: {:,d}(relationships: {:,d} - attributes: {:,d})\n"
                "number of time instances: {:,d}".format(self.num_nodes,
                                                      self.num_entities,
                                                      self.num_values,
                                                      self.num_relations,
                                                      self.num_relationship_relations,
                                                      self.num_attribute_relations,
                                                      self.num_edges,
                                                        self.num_relationship_edges,
                                                        self.num_attribute_edges,
                                                      self.num_time_instances))

    def is_entity_id(self, entity_id):
        return entity_id in self.__entity_ids

    def get_num_entities(self, time_instance):
        time_graph = self._get_time_graph(time_instance)

        count = 0
        for node_id in time_graph.nodes():
            if self.is_entity_id(node_id):
                count += 1

        return count

    def _get_time_graph(self, time_instance):
        time_graph = self.__graphs.get(time_instance, 0)

        return time_graph

    def get_num_edges(self, time_instance):
        time_graph = self._get_time_graph(time_instance)

        return time_graph.number_of_edges()

    def get_edges_iter(self, time_instance):
        time_graph = self._get_time_graph(time_instance)

        for node1_id, node2_id, relation_id in time_graph.edges(keys=True):
            yield node1_id, relation_id, node2_id

    def get_domain_value_ids(self, relation_id):
        return self.__attribute_domains[relation_id]

    def get_relation_edges_iter(self, time_instance, relation_id):
        time_graph = self._get_time_graph(time_instance)

        for node1_id, node2_id, cur_relation_id in time_graph.edges(keys=True):
            if relation_id == cur_relation_id:
                yield node1_id, cur_relation_id, node2_id

    def get_entities_iter(self, time_instance):
        time_graph = self._get_time_graph(time_instance)

        for node_id in time_graph.nodes():
            if self.is_entity_id(node_id):
                yield node_id

    def has_edge_id(self, node1_id, relation_id, node2_id, time_instance):
        time_graph = self._get_time_graph(time_instance)

        return time_graph.has_edge(u=node1_id, v=node2_id, key=relation_id)

    def has_node_id(self, node_id, time_instance):
        time_graph = self._get_time_graph(time_instance)

        return time_graph.has_node(node_id)


    def __add_edge_from_id(self, node1_id, relation_id, node2_id, time):
        if time not in self.__graphs:
            self.__graphs[time] = nx.MultiDiGraph()

        if self.has_edge_id(node1_id, relation_id, node2_id, time):
            return

        if self.is_attribute_relation_id(relation_id):
            self.__num_attribute_edges += 1
        else:
            self.__num_relationship_edges += 1

        time_graph = self.__graphs[time]
        time_graph.add_edge(node1_id, node2_id, key=relation_id)

    def add_relationship_edge_from_id(self, entity1_id, relation_id, entity2_id, time):
        self.__entity_ids.add(entity1_id)
        self.__entity_ids.add(entity2_id)
        self.__relationship_relation_ids.add(relation_id)

        self.__add_edge_from_id(entity1_id, relation_id, entity2_id, time)

    def add_attribute_domain_from_id(self, relation_id, value_id):
        domain = self.__attribute_domains.get(relation_id)

        if domain is None:
            domain = set()
            self.__attribute_domains[relation_id] = domain

        domain.add(value_id)

    def add_attribute_edge_from_id(self, entity_id, relation_id, value_id, time):
        self.__entity_ids.add(entity_id)
        self.__value_ids.add(value_id)
        self.__attribute_relation_ids.add(relation_id)

        self.add_attribute_domain_from_id(relation_id, value_id)

        self.__add_edge_from_id(entity_id, relation_id, value_id, time)

    def get_node_id(self, name):
        entity_id = _get_item_id_from_name(self.__node2id, name)
        return entity_id

    def get_relation_type_from_id(self, relation_id):
        if self.is_attribute_relation_id(relation_id):
            return ATTRIBUTE_RELATION_TYPE
        elif self.is_relationship_relation_id(relation_id):
            return RELATIONSHIP_RELATION_TYPE
        else:
            return None

    def get_relation_raw_name(self, name, relation_type):
        return "{}_{}".format(name, relation_type)

    def get_relation_id(self, name, relation_type):
        raw_name = self.get_relation_raw_name(name, relation_type)
        relation_id = _get_item_id_from_name(self.__relation2id, raw_name)
        return relation_id

    def get_attribute_relation_raw_name(self, name):
        raw_name = self.get_relation_raw_name(name, ATTRIBUTE_RELATION_TYPE)
        logger.debug("raw name of {} is {}".format(name, raw_name))
        return raw_name


    def add_relationship_edge(self, entity1_name, relation_name, entity2_name, time):
        entity1_id = self.get_node_id(entity1_name)
        entity2_id = self.get_node_id(entity2_name)
        relation_id = self.get_relation_id(relation_name, RELATIONSHIP_RELATION_TYPE)

        self.add_relationship_edge_from_id(
            entity1_id, relation_id, entity2_id, time)

    def add_attribute_edge(self, entity_name, relation_name, value_name, time):
        entity_id = self.get_node_id(entity_name)
        value_id = self.get_node_id(value_name)
        relation_id = self.get_relation_id(relation_name, ATTRIBUTE_RELATION_TYPE)

        self.add_attribute_edge_from_id(entity_id, relation_id, value_id, time)

    def is_attribute_relation_id(self, relation_id):
        return relation_id in self.__attribute_relation_ids

    def is_attribute_relation_name(self, relation_name):
        logger.debug("relations: {}".format(self.__relation2id))
        return self.get_attribute_relation_raw_name(relation_name) in self.__relation2id

    def is_relationship_relation_id(self, relation_id):
        return relation_id in self.__relationship_relation_ids

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        self.__save_index(path)
        self.__save_domains(path)
        self.__save_edges(path)

    def __save_domains(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        domains_path = os.path.join(path, 'domains.txt')

        with open(domains_path, 'w') as f:
            for relation_id, domain in self.__attribute_domains.items():
                domain_str = ""

                for value_id in domain:
                    domain_str += "{},".format(value_id)

                line = "{}:{}\n".format(relation_id, domain_str[:-1])
                f.write(line)


    def __save_index(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        entityidx_path = os.path.join(path, 'entities.idx')
        valueidx_path = os.path.join(path, 'values.idx')
        attrsidx_path = os.path.join(path, 'attrs.idx')
        relsidx_path = os.path.join(path, 'rels.idx')

        save_index(valueidx_path, self.__value_ids, self.__node2id)
        save_index(entityidx_path, self.__entity_ids, self.__node2id)
        save_index(attrsidx_path, self.__attribute_relation_ids,
                    self.__relation2id)
        save_index(relsidx_path, self.__relationship_relation_ids,
                    self.__relation2id)

    def __save_edges(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        logger.debug("writing edges to: {}".format(path))
        rel_info_path = os.path.join(path, 'rels.edges')
        attr_info_path = os.path.join(path, 'attrs.edges')

        # save metadata
        rel_info_file = open(rel_info_path, 'w')
        attr_info_file = open(attr_info_path, 'w')

        for t, graph in self.__graphs.items():
            logger.debug("writing at time {}".format(t))
            for entity1_id, entity2_id, relation_id in graph.edges(keys=True):
                line = '{},{},{},{}\n'.format(
                    entity1_id, relation_id, entity2_id, t)
                # logger.debug(line)
                if self.is_attribute_relation_id(relation_id):
                    # logger.debug('write to attr file')
                    current_file = attr_info_file
                else:
                    # logger.debug('write to rel file')
                    current_file = rel_info_file

                current_file.write(line)

        rel_info_file.close()
        attr_info_file.close()

    @staticmethod
    def from_raw_file(data_dir, args):
        raise NotImplementedError("Not implemented")

    @staticmethod
    def from_raw_graph_output(path):
        graph = DynamicGraph()

        node2id, relation2id, user_ids, value_ids, attribute_relation_ids, relationship_relation_ids = read_index_data(path)

        # logger.info(relation2id)
        # raise Exception()
        graph.__node2id = node2id
        graph.__relation2id = relation2id
        graph.__entity_ids = user_ids
        graph.__value_ids = value_ids
        graph.__attribute_relation_ids = attribute_relation_ids
        graph.__relationship_relation_ids = relationship_relation_ids

        graph.__attribute_domains = read_domains_data(path)

        read_edges(path, graph)

        return graph

    def generate_subgraph_from_time_instances(self, time_instances, sensitive_attr, entity2extra_svals):
        # add indexes
        result_graph = SubGraph.from_index_data(self.__node2id, self.__relation2id, self.__relationship_relation_ids, self.__attribute_relation_ids, self.__attribute_domains)

        result_graph.sensitive_name = sensitive_attr

        having_sattr_entities = set()

        # add edges
        for t in time_instances:
            current_graph = self.__graphs[t]

            for node1_id, node2_id, relation_id in current_graph.edges(keys=True):
                if result_graph.sensitive_attr_id == relation_id:
                    result_graph.add_sensitive_value_id(node1_id, relation_id, node2_id)
                    having_sattr_entities.add(node1_id)
                elif self.is_attribute_relation_id(relation_id):
                    result_graph.add_attribute_edge_from_id(node1_id, relation_id, node2_id)
                else:
                    result_graph.add_relationship_edge_from_id(node1_id, relation_id, node2_id)

        sattr_id = self.__relation2id[self.get_relation_raw_name(sensitive_attr, ATTRIBUTE_RELATION_TYPE)]
        for entity_id, sval_ids in entity2extra_svals.items():
            for sval_id in sval_ids:
                result_graph.add_sensitive_value_id(entity_id, sattr_id, sval_id)

        return result_graph

    def copy(self):
        graph = DynamicGraph()
        graph.__graphs = self.__graphs.copy()
        graph.__node2id = self.__node2id.copy()
        graph.__relation2id = self.__relation2id.copy()
        graph.__relationship_relation_ids = self.__relationship_relation_ids.copy()
        graph.__attribute_relation_ids = self.__attribute_relation_ids.copy()
        graph.__entity_ids = self.__entity_ids.copy()
        graph.__value_ids = self.__value_ids.copy()
        graph.__attribute_domains = self.__attribute_domains.copy()
        graph.__time_unit = self.__time_unit

        return graph

def read_domains_data(path):
    domain_path = os.path.join(path, 'domains.txt')

    domains_data = {}
    with open(domain_path, 'r') as f:
        for line in f:
            relation_id, domain_str = line.strip().split(':')
            domain_value_ids = set(map(int, domain_str.split(',')))
            domains_data[int(relation_id)] = domain_value_ids

    return domains_data

def read_index_data(path):
    users_idx_path = os.path.join(path, 'entities.idx')
    values_idx_path = os.path.join(path, 'values.idx')
    attrs_idx_path = os.path.join(path, 'attrs.idx')
    rels_idx_path = os.path.join(path, 'rels.idx')

    node2id = read_index_file([users_idx_path])
    user_ids = set(node2id.values())

    value2id = read_index_file([values_idx_path])
    value_ids = set(value2id.values())

    # raise Exception("{} {}".format(user_ids, value_ids))
    node2id.update(value2id)

    relation2id = read_index_file([attrs_idx_path])
    attribute_relation_ids = set(relation2id.values())

    relationship2id = read_index_file([rels_idx_path])
    relationship_relation_ids = set(relationship2id.values())

    relation2id.update(relationship2id)

    # raise Exception("{} \n{} \n{} \n{} \n{} \n{}".format(node2id, relation2id, user_ids, value_ids, attribute_relation_ids, relationship_relation_ids))
    return node2id, relation2id, user_ids, value_ids, attribute_relation_ids, relationship_relation_ids


def read_index_file(file_paths):
    result = {}

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                # logger.debug('line: {}'.format(line))
                splits = line.rstrip().split(',')
                name = ','.join(splits[0:-1])
                idx = int(splits[-1])

                # logger.debug('name = {} - idx = {}'.format(name, idx))
                # name, idx = splits[0], int(splits[1])
                result[name] = idx

    return result


def read_edges(path, graph):
    attr_edges_path = os.path.join(path, 'attrs.edges')
    rel_edges_path = os.path.join(path, 'rels.edges')

    read_edges_file(attr_edges_path, graph, lambda graph, node1_id, relation_id,
                     node2_id, t: graph.add_attribute_edge_from_id(node1_id, relation_id, node2_id, t))
    read_edges_file(rel_edges_path, graph, lambda graph, node1_id, relation_id, node2_id,
                     t: graph.add_relationship_edge_from_id(node1_id, relation_id, node2_id, t))

def read_edges_file(path, graph, edge_addition_fn):
    with open(path, 'r') as f:
        for line in f:
            splits = line.strip().split(',')
            entity_id, relation_id, node_id, t = int(splits[0]), int(
                splits[1]), int(splits[2]), int(splits[3])

            edge_addition_fn(graph, entity_id, relation_id, node_id, t)


def _get_item_id_from_name(name2id, name):
    item_id = name2id.get(name, None)

    if item_id is None:
        item_id = len(name2id)
        name2id[name] = item_id

    return item_id


def save_index(path, ids, name2id):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'w') as f:
        for id in ids:
            name = _get_name_from_id(id, name2id)
            f.write("{},{}\n".format(name, id))


def _get_name_from_id(id, name2id):
    for item_name, item_id in name2id.items():
        if id == item_id:
            return item_name

    return None
