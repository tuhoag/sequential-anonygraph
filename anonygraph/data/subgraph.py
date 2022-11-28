import os
import logging
import networkx as nx

import anonygraph.utils.path as putils
from anonygraph import data
from anonygraph.constants import *

logger = logging.getLogger(__name__)


class SubGraph(object):
    def __init__(self):
        # self.__time_instance = time_instance
        self.__graph = nx.MultiDiGraph()

        # take from dynamic graph
        self.__node2id = {}
        self.__relation2id = {}
        self.__relationship_relation_ids = set()
        self.__attribute_relation_ids = set()
        self.__attribute_domains = {}

        # nodes exist in this subgraph
        self.__entity_ids = set()
        self.__value_ids = set()

        # sensitive values
        self.__sensitive_attr_id = None
        self.__entity2sensitive_vals = {}

    @property
    def entity2svals(self):
        return self.__entity2sensitive_vals

    @property
    def raw_graph(self):
        return self.__graph

    @property
    def num_nodes(self):
        return self.num_entities + self.num_values

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
    def num_attribute_relations(self):
        return len(self.__attribute_relation_ids)

    @property
    def num_relationship_relations(self):
        return len(self.__relationship_relation_ids)

    @property
    def num_edges(self):
        return self.__graph.number_of_edges()

    def __str__(self):
        return ("number of nodes: {} (entities: {} - values: {})\n"
            "number of relations: {} (relationships: {} - attributes: {})\n"
            "number of edges: {}\n".format(self.num_nodes,
                                                    self.num_entities,
                                                    self.num_values,
                                                    self.num_relations,
                                                    self.num_relationship_relations,
                                                    self.num_attribute_relations,
                                                    self.num_edges))


    @property
    def entity_ids(self):
        return self.__entity_ids

    def is_entity_id(self, entity_id):
        return entity_id in self.__entity_ids

    @property
    def relationship_relation_ids(self):
        return self.__relationship_relation_ids

    @property
    def attribute_relation_ids(self):
        return self.__attribute_relation_ids

    @property
    def sensitive_raw_name(self):
        for relation_name, relation_id in self.__relation2id.items():
            if relation_id == self.__sensitive_attr_id:
                return relation_name

        return None

    @property
    def sensitive_name(self):
        return self.sensitive_raw_name.split("_")[0]

    #  @x.setter
    # def x(self, x):
    #     if x < 0:
    #         self.__x = 0
    #     elif x > 1000:
    #         self.__x = 1000
    #     else:
    #         self.__x = x

    @sensitive_name.setter
    def sensitive_name(self, name):
        raw_name = "{}_{}".format(name, ATTRIBUTE_RELATION_TYPE)
        # logger.info("node names: {}".format(self.__node2id.keys()))
        # logger.info("relation names: {}".format(self.__relation2id.keys()))
        self.__sensitive_attr_id = self.__relation2id[raw_name]

    @property
    def sensitive_attr_id(self):
        return self.__sensitive_attr_id

    @staticmethod
    def from_index_data(node2id, relation2id, relationship_relation_ids, attribute_relation_ids, attribute_domains):
        graph = SubGraph()

        graph.__node2id = node2id
        graph.__relation2id = relation2id
        graph.__relationship_relation_ids = relationship_relation_ids
        graph.__attribute_relation_ids = attribute_relation_ids
        graph.__attribute_domains = attribute_domains

        return graph

    @staticmethod
    def from_index_and_edges_data(node2id, relation2id, relationship_relation_ids, attribute_relation_ids, attribute_domains, attribute_edges, relationship_edges, sensitive_attr_id, entity2sensitive_vals):
        """Load Subgraph from index of dynamic graph and its edges data.

        Arguments:
            node2id {[type]} -- [description]
            relation2id {[type]} -- [description]
            relationship_relation_ids {[type]} -- [description]
            attribute_relation_ids {[type]} -- [description]
            attribute_domains {[type]} -- [description]
            attribute_edges {[type]} -- [description]
            relationship_edges {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        graph = SubGraph.from_index_data(
            node2id, relation2id, relationship_relation_ids, attribute_relation_ids, attribute_domains)
        graph.__sensitive_attr_id = sensitive_attr_id
        graph.__entity2sensitive_vals = entity2sensitive_vals

        for node1_id, relation_id, node2_id in attribute_edges:
            # logger.debug("add attribute edge: {}, {}, {}".format(node1_id, relation_id, node2_id))
            graph.add_attribute_edge_from_id(node1_id, relation_id, node2_id)
            # if relation_id == 11:
            #     raise Exception()
            # logger.debug(graph)
            # logger.debug("entities: {} - values: {}".format(graph.__entity_ids, graph.__value_ids))


        for node1_id, relation_id, node2_id in relationship_edges:
            # logger.debug("add relationship edge: {}, {}, {}".format(node1_id, relation_id, node2_id))
            graph.add_relationship_edge_from_id(
                node1_id, relation_id, node2_id)

            # logger.debug("entities: {} - values: {}".format(graph.__entity_ids, graph.__value_ids))
            # logger.debug(graph)

        return graph

    def __add_edge_from_id(self, node1_id, relation_id, node2_id):
        self.__graph.add_edge(node1_id, node2_id, key=relation_id)

    def remove_edge_from_id(self, node1_id, relation_id, node2_id):
        self.__graph.remove_edge(node1_id, node2_id, key=relation_id)

    def add_attribute_edge_from_id(self, entity_id, relation_id, value_id):
        self.__entity_ids.add(entity_id)
        self.__value_ids.add(value_id)
        self.__attribute_relation_ids.add(relation_id)

        self.__add_edge_from_id(entity_id, relation_id, value_id)

    def remove_entity_id(self, entity_id):
        self.__entity_ids.remove(entity_id)
        self.__graph.remove_node(entity_id)

    def add_relationship_edge_from_id(self, entity1_id, relation_id, entity2_id):
        self.__entity_ids.add(entity1_id)
        self.__entity_ids.add(entity2_id)
        self.__relationship_relation_ids.add(relation_id)

        self.__add_edge_from_id(entity1_id, relation_id, entity2_id)

    def is_attribute_relation_id(self, relation_id):
        return relation_id in self.__attribute_relation_ids

    def is_relationship_relation_id(self, relation_id):
        return relation_id in self.__relationship_relation_ids

    def to_edge_files(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        rel_info_path = os.path.join(path, "rels.edges")
        attr_info_path = os.path.join(path, "attrs.edges")

        with open(rel_info_path, "w") as rel_info_file, open(attr_info_path, "w") as attr_info_file:
            for entity1_id, entity2_id, relation_id in self.__graph.edges(keys=True):
                line = "{},{},{}\n".format(
                    entity1_id, relation_id, entity2_id)
                # logger.debug(line)
                if self.is_attribute_relation_id(relation_id):
                    # logger.debug("write to attr file")
                    if relation_id == self.__sensitive_attr_id:
                        raise Exception("There is an sensitive edge which is in edges data: {},{},{}".format(entity1_id, relation_id, entity2_id))

                    current_file = attr_info_file
                elif self.is_relationship_relation_id(relation_id):
                    # logger.debug("write to rel file")
                    current_file = rel_info_file

                current_file.write(line)

        sensitive_vals_path = os.path.join(path, "sensitive.vals")
        with open(sensitive_vals_path, "w") as sen_file:
            sen_file.write("{}\n".format(self.__sensitive_attr_id))

            for entity_id, val_ids in self.__entity2sensitive_vals.items():
                for val_id in val_ids:
                    sen_file.write("{},{}\n".format(entity_id, val_id))

    def get_domain_value_ids(self, relation_id):
        return self.__attribute_domains[relation_id]

    def get_attribute_edges_of_entity_id(self, entity_id):
        if self.is_entity_id(entity_id):
            for _, value_id, relation_id in self.__graph.out_edges(entity_id, keys=True):
                if self.is_attribute_relation_id(relation_id):
                    yield entity_id, relation_id, value_id

    def get_out_relationship_edges_of_entity_id(self, entity1_id):
        if self.is_entity_id(entity1_id):
            for _, entity2_id, relation_id in self.__graph.out_edges(entity1_id, keys=True):
                if not self.is_attribute_relation_id(relation_id):
                    yield entity1_id, relation_id, entity2_id

    def get_in_relationship_edges_of_entity_id(self, entity_id):
        if self.is_entity_id(entity_id):
            for entity2_id, _, relation_id in self.__graph.in_edges(entity_id, keys=True):
                yield entity2_id, relation_id, entity_id

    def get_edges_iter(self):
        for node1_id, node2_id, relation_id in self.__graph.edges(keys=True):
            yield node1_id, relation_id, node2_id

    def get_relationship_edges_iter(self):
        for node1_id, node2_id, relation_id in self.__graph.edges(keys=True):
            if self.is_relationship_relation_id(relation_id):
                yield node1_id, relation_id, node2_id

    def get_edges_iter_of_relation_id(self, relation_id):
        for node1_id, node2_id, current_relation_id in self.__graph.edges(keys=True):
            if relation_id == current_relation_id:
                yield node1_id, current_relation_id, node2_id

    def get_out_edges_iter(self, entity_id):
        for node1_id, node2_id, relation_id in self.__graph.edges(nbunch=entity_id, keys=True):
            yield node1_id, relation_id, node2_id

    def get_in_edges_iter(self, entity_id):
        for node1_id, node2_id, relation_id in self.__graph.in_edges(nbunch=entity_id, keys=True):
            yield node1_id, relation_id, node2_id

    def get_num_domain_value_ids_from_relation_id(self, relation_id):
        # logger.debug("relation id: {} - domain: {}".format(relation_id, self.__attribute_domains))
        domain_value_ids = self.__attribute_domains.get(relation_id)
        return len(domain_value_ids)

    def is_edge_existed(self, node1_id, relation_id, node2_id):
        return self.__graph.has_edge(node1_id, node2_id, relation_id)

    # def remove_edges_of_attribute_name(self, attribute_name):
    #     # get relation id
    #     attr_id = self.__relation2id[attribute_name]
    #     if not self.is_attribute_relation_id(attr_id):
    #         raise Exception("{} is not attribute".format(attribute_name))

    #     for entity_id in self.__entity_ids:
    #         attr_edges = self.get_attribute_edges_of_entity_id(entity_id)

    #         for _, relation_id, value_id in attr_edges:
    #             if relation_id == attr_id:

    def add_sensitive_value_id(self, entity_id, relation_id, value_id):
        if relation_id != self.__sensitive_attr_id:
            raise Exception("relation_id '{}' is not the sensitive attr id '{}'".format(relation_id, self.__sensitive_attr_id))

        entity_sensitive_values = self.__entity2sensitive_vals.get(entity_id)

        if entity_sensitive_values is None:
            entity_sensitive_values = set()
            self.__entity2sensitive_vals[entity_id] = entity_sensitive_values

        entity_sensitive_values.add(value_id)

    def get_sensitive_value_id(self, entity_id):
        sval_ids  = self.__entity2sensitive_vals.get(entity_id)

        return sval_ids

def get_edges_iter(path):
    with open(path, "r") as file:
        for line in file:
            entity1_id, relation_id, entity2_id = list(map(int, line.strip().split(",")))
            yield entity1_id, relation_id, entity2_id
