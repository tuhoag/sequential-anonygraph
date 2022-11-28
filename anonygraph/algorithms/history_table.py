import os
import logging
import json
import jsonpickle
from sortedcontainers import SortedSet, SortedDict

import anonygraph.info_loss.info as info
import anonygraph.constants as constants

logger = logging.getLogger(__name__)

class HistoryTable(object):
    def __init__(self):
        self.__data = {}
        self.__time_insts = SortedSet()
        self.__signatures = {}
        self.__sensitive_attr = None

    @property
    def first_time_instance(self):
        if len(self.__time_insts) == 0:
            return -1

        return self.__time_insts[0]

    @property
    def last_time_instance(self):
        if len(self.__time_insts) == 0:
            return -1

        return self.__time_insts[-1]

    @property
    def num_time_instances(self):
        return len(self.__time_insts)

    @property
    def time_instances(self):
        return self.__time_insts

    def remove_at(self, time_inst):
        entity_ids = list(self.__data.keys())

        for entity_id in entity_ids:
            history_entry = self.__data[entity_id]
            logger.debug('remove entry of entity: {}'.format(entity_id))
            logger.debug('before: {}'.format(self.__data[entity_id]))
            history_entry.pop(time_inst, None)
            logger.debug('after: {}'.format(self.__data[entity_id]))

            if len(history_entry) == 0:
                self.__data.pop(entity_id)
                self.__signatures.pop(entity_id)

        logger.debug('time before: {}'.format(self.__time_insts))
        self.__time_insts.discard(time_inst)
        logger.debug('time after: {}'.format(self.__time_insts))

    def remove_first_history(self):
        first_time_inst = self.first_time_instance

        if first_time_inst > -1:
            self.remove_at(first_time_inst)

    def add_new_entity_info(self, entity_id, attr_info, out_info, in_info, time_inst, signature={}):
        signature_entry = self.__signatures.get(entity_id)
        if signature_entry is None:
            self.__signatures[entity_id] = signature
        elif signature_entry != signature:
            logger.error("Signature is violated (previous signature: '{}' - new signature: '{}'".format(signature_entry, signature))
            raise Exception("Signature is violated (previous signature: '{}' - new signature: '{}'".format(signature_entry, signature))

        self.__signatures[entity_id] = signature

        history_entry = self.__data.get(entity_id)
        if history_entry is None:
            history_entry = {}
            self.__data[entity_id] = history_entry

        history_entry[time_inst] = {
            'out': out_info,
            'in': in_info,
            'attr': attr_info,
        }

        self.__time_insts.add(time_inst)


    def add_empty_new_entry(self, entity_ids, time_inst):
        for entity_id in entity_ids:
            signature = self.__signatures[entity_id]
            self.add_new_entity_info(entity_id, {}, {}, {}, time_inst, signature)

    @property
    def entity_ids(self):
        return set(self.__data.keys())

    def get_previous_time_instance(self, time_inst):
        if len(self.__time_insts) == 0:
            return -1

        logger.debug(self.__time_insts)
        logger.debug(time_inst)
        t_index = self.__time_insts.index(time_inst)

        if t_index == 0:
            return -1

        t_previous = self.__time_insts[t_index - 1]
        return self.__time_insts[t_previous]

    def get_entity_ids_in_recent_time_instance(self):
        recent_time_inst = self.last_time_instance
        logger.debug("recent time inst: {}".format(recent_time_inst))
        return self.get_entity_ids_in_time_instance(recent_time_inst)

    def get_entity_ids_in_previous_time_instance(self, time_inst):
        previous_time_inst = self.get_previous_time_instance(time_inst)
        return self.get_entity_ids_in_time_instance(previous_time_inst)

    def get_entity_ids_in_time_instance(self, time_inst):
        result = set()

        for entity_id, entity_history in self.__data.items():
            if time_inst in entity_history:
                result.add(entity_id)
            # logger.debug(result)

        return result

    def add_new_entry_from_clusters(self, clusters, subgraph, fake_entity_manager, time_inst):
        remaining_entity_ids = self.entity_ids
        logger.debug('all entities: {}'.format(remaining_entity_ids))

        for cluster in clusters:
            cluster_entity_ids = cluster.to_list()
            logger.debug("cluster entity ids: {}".format(cluster_entity_ids))

            # get attr info
            generated_attr_info, _ = info.get_generalized_attribute_info(subgraph, cluster_entity_ids)

            # get out info
            generated_out_info, _ = info.get_generalized_degree_info(subgraph, cluster_entity_ids, 'out')

            # get in info
            generated_in_info, _ = info.get_generalized_degree_info(subgraph, cluster_entity_ids, 'in')

            # signature
            generalized_signature_info = info.get_generalized_signature_info(subgraph, fake_entity_manager, cluster_entity_ids)

            # logger.debug(generated_attr_info)
            logger.debug("signature: {}".format(generalized_signature_info))

            # if len(cluster_entity_ids) > 0:

            #     raise Exception()

            # add to history
            for entity_id in cluster_entity_ids:
                # logger.debug('adding entity {} at t {} info: {}'.format(entity_id, time_inst, generated_info))


                self.add_new_entity_info(entity_id, generated_attr_info, generated_out_info, generated_in_info, time_inst, generalized_signature_info)

                logger.debug('remove entity: {}'.format(entity_id))
                remaining_entity_ids.discard(entity_id)

        self.add_empty_new_entry(remaining_entity_ids, time_inst)

    def add_new_entry_from_subgraph(self, subgraph, time_inst):
        logger.debug("entity_ids: {}".format(subgraph.entity_ids))
        logger.debug("entity2svals: {}".format(subgraph.entity2svals))

        for entity_id in subgraph.entity_ids:
            logger.debug("entity_id: {}".format(entity_id))

            attr_info = info.get_attribute_info(subgraph, entity_id)
            out_info = info.get_degree_info(subgraph, entity_id, 'out')
            in_info = info.get_degree_info(subgraph, entity_id, 'in')
            signature = info.get_signature_info(subgraph, entity_id)

            logger.debug("signature: {}".format(signature))
            # raise Exception()
            self.add_new_entity_info(entity_id, attr_info, out_info, in_info, time_inst, signature)


    def get_history_key(self, entity_id):
        """This function generates a history key of an entity. This key will be used to merge users whose historical information (attr values and degrees) and signature are identical. If he/she is a new user, his/her key is EMPTY_HISTORY_KEY.

        Args:
            entity_id ([type]): [description]

        Returns:
            [type]: [description]
        """

        history_entry = self.__data.get(entity_id)

        if history_entry is not None:
            signature_str = str(self.get_signature(entity_id))
            # history_entry_str = "{}:{}".format(history_entry,signature_str)



            history_entry_str = "{}".format(history_entry)
        else:
            history_entry_str = constants.EMPTY_HISTORY_KEY

        logger.debug('entity: {} - history: {} - str: {}'.format(entity_id, history_entry, history_entry_str))

        return history_entry_str


    def to_file(self, path):
        logger.debug('saving history to path: {}'.format(path))
        logger.debug(self.__time_insts)
        logger.debug(self.__data)
        # raise Exception()
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, 'w+') as f:
            histories_data = {}
            signatures_data = {}

            data = {
                "histories": histories_data,
                "signatures": signatures_data,
            }

            for entity_id, history_entry in self.__data.items():
                logger.debug('entity: {} - history entry: {}'.format(entity_id, history_entry))
                entity_entry = {}

                for t, time_entry in history_entry.items():
                    logger.debug('t: {} - time_entry: {}'.format(t, time_entry))
                    entry = {
                        'out': time_entry['out'],
                        'in': time_entry['in'],
                        'attr': {}
                    }

                    for relation_id, value_ids in time_entry['attr'].items():
                        entry['attr'][relation_id] = list(value_ids)

                    entity_entry[t] = entry

                histories_data[entity_id] = entity_entry

                signature = self.get_signature(entity_id)
                # logger.debug(self.__signatures)
                # logger.debug(signature)
                # raise Exception()
                signatures_data[entity_id] = list(signature)

            logger.debug(data)
            json.dump(data, f, indent=4)

    @staticmethod
    def from_file(path):
        history = HistoryTable()

        with open(path, 'r') as f:
            data = json.load(f)
            logger.debug(data)

            signatures_data = data["signatures"]
            entity2sign = {}

            for entity_id_str, signature_entry in signatures_data.items():
                entity_id = int(entity_id_str)
                entity2sign[entity_id] = set(map(int, signature_entry))
                # history.__signatures[entity_id] =

            # logger.debug(history)
            # raise Exception()
            histories_data = data["histories"]

            for entity_id_str, history_entry in histories_data.items():
                entity_id = int(entity_id_str)

                logger.debug('entity: {} - history entry: {}'.format(entity_id, history_entry))
                # raise Exception()
                for t_str, time_entry in history_entry.items():
                    t = int(t_str)
                    logger.debug('t: {} - time_entry: {}'.format(t, time_entry))

                    attr_info = SortedDict()
                    out_info = SortedDict()
                    in_info = SortedDict()

                    for relation_id, vals in time_entry['attr'].items():
                        attr_info[int(relation_id)] = SortedSet(map(int, vals))

                    for relation_id, degree in time_entry['out'].items():
                        out_info[int(relation_id)] = int(degree)

                    for relation_id, degree in time_entry['in'].items():
                        in_info[int(relation_id)] = int(degree)

                    signature = entity2sign[entity_id]
                    # logger.debug('entity info: {}'.format(entity_info))
                    history.add_new_entity_info(entity_id, attr_info, out_info, in_info, t, signature)

            logger.debug('loaded data: {}'.format(history.__data))
                    # raise Exception()
            logger.debug('loaded time instances: {}'.format(history.__time_insts))
        return history

    def __str__(self):
        return str(self.__data)

    def __repr__(self):
        return str(self)

    def get_signature(self, entity_id):
        return self.__signatures.get(entity_id)

def get_dict_key():

    pass