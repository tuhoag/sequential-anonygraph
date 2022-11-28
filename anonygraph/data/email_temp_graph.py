import tqdm
import networkx as nx
import logging
import os

from .dynamic_graph import DynamicGraph

logger = logging.getLogger(__name__)

GRAPH_NAME = 'email'
USER_RELATION_NAME = 'sent'
ATTR_RELATION_NAME = 'dept'


def get_user_name(id, dept_id):
    return "dept{}:user{}".format(dept_id, id)

def get_dept_name(id):
    return "dept{}".format(id)

def load_users_relationship(graph, file_path, dept_id, entity2dept, t2entity):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = list(map(lambda line: line.rstrip().split(' '), lines))

        # entity_ids = set()
        time_insts = set()
        for u, v, t in lines:
            u_name = get_user_name(u, dept_id)
            v_name = get_user_name(v, dept_id)
            time_insts.add(t)
            dept_name = get_dept_name(dept_id)

            graph.add_relationship_edge(u_name, USER_RELATION_NAME, v_name, t)

            entity2dept[u_name] = dept_name
            entity2dept[v_name] = dept_name

            entities = t2entity.get(t, set())
            entities.add(u_name)
            entities.add(v_name)
            t2entity[t] = entities

            # u_key = "{}_{}".format(u_name, t)
            # v_key = "{}_{}".format(v_name, t)
            # if u_key in entity_ids:
            #     count_before = graph.get_num_edges(t)

            # graph.add_attribute_edge(u_name, ATTR_RELATION_NAME, dept_name, t)

            # if u_key in entity_ids:
            #     count_after = graph.get_num_edges(t)

            #     assert count_before == count_after, "before: {} - after:{}".format(count_before, count_after)

            # if v_key in entity_ids:
            #     count_v_before=graph.get_num_edges(t)

            # graph.add_attribute_edge(v_name, ATTR_RELATION_NAME, dept_name, t)

            # if v_key in entity_ids:
            #     count_v_after = graph.get_num_edges(t)
            #     assert count_v_before == count_v_after, "before: {} - after: {}".format(count_v_before, count_v_after)

            # entity_ids.add(u_key)
            # entity_ids.add(v_key)

def add_attribute_relationships(graph, entity2dept, t2entity):
    time_ints = graph.time_instances

    for t, entities in tqdm.tqdm(t2entity.items()):
        for entity in entities:
            dept_name = entity2dept[entity]
            graph.add_attribute_edge(entity, ATTR_RELATION_NAME, dept_name, t)
    # for entity, dept_name in tqdm.tqdm(entity2dept.items()):
    #     for t in time_ints:
    #         graph.add_attribute_edge(entity, dept_name, dept_name, t)

class EmailTempGraph(DynamicGraph):
    def __init__(self):
        super().__init__()

    @staticmethod
    def from_raw_file(data_dir, args):
        # load attributes
        # load relationships
        graph = EmailTempGraph()
        entity2dept = {}
        t2entity = {}


        user_relationship_path = os.path.join(data_dir, 'email-Eu-core-temporal.txt')
        load_users_relationship(graph, user_relationship_path, 0, entity2dept, t2entity)

        logger.debug("{} entities2dept: of dept 0".format(len(entity2dept)))
        logger.debug("{} t2entity: of dept 0".format(len(t2entity)))

        logger.info("adding users relationships")
        for dept_id in range(1,5):
            user_relationship_path = os.path.join(data_dir, 'email-Eu-core-temporal-Dept{}.txt'.format(dept_id))
            load_users_relationship(graph, user_relationship_path, dept_id, entity2dept, t2entity)

        logger.info("adding attribute relationships")
        add_attribute_relationships(graph, entity2dept, t2entity)

        return graph
