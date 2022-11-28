import os
import logging

from anonygraph.data.dynamic_graph import read_index_file, save_index

logger = logging.getLogger(__name__)

class FakeEntityManager(object):
    def __init__(self):
        self.entity2id = {}
        self.fake_entity2id = {}
        self.current_max_id = -1
        self.fake_entity_ids = set()
        self.fake_entity2sval = {}

    def update_current_max_id(self):
        if len(self.fake_entity2sval) == 0:
            # get max
            self.current_max_id = max(map(int, self.entity2id.values()))
        else:
            self.current_max_id = max(map(int, self.fake_entity2sval.keys()))

        logger.debug("self.entity2id: {}".format(self.entity2id))
        logger.debug("self.fake_entity2sval: {}".format(self.fake_entity2sval))
        logger.debug("self.current_max_id: {}".format(self.current_max_id))
        # raise Exception()

    def get_sensitive_value_id(self, fake_entity_id):
        return self.fake_entity2sval[fake_entity_id]

    def update_fake_entity_ids(self):
        logger.debug("self.entity2id: {}".format(self.entity2id))
        logger.debug("self.fake_entity2id: {}".format(self.fake_entity2id))
        logger.debug("self.current_max_id: {}".format(self.current_max_id))
        logger.debug("self.fake_entity_ids: {}".format(self.fake_entity_ids))
        logger.debug("self.fake_entity2sval: {}".format(self.fake_entity2sval))

        # raise Exception()
        self.fake_entity_ids = set(self.fake_entity2sval.keys())
        logger.debug("updated self.fake_entity_ids: {}".format(self.fake_entity_ids))

    def create_new_fake_entity(self, svalue_id):
        self.current_max_id += 1

        fake_entity_id = self.current_max_id
        fake_entity_name = "fake_{}".format(fake_entity_id)
        # self.fake_entity2id[fake_entity_id] = fake_entity_name
        self.fake_entity2sval[fake_entity_id] = svalue_id

        # self.fake_entity2id[fake_entity_id] = fake_entity_name{
        #     "name": fake_entity_name,
        #     "sid": svalue_id,
        # }

        return fake_entity_id

    # def add_existed_fake_entity_id(self, fake_entity_id, sval_id):
    #     self.fake_entity2sval[fake_entity_id] = sval_id
    #     self.fake_entity_ids.add(fake_entity_id)


    def to_file(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            logger.info("created folder: {}".format(os.path.dirname(path)))
        else:
            logger.info("folder is existed at: {}".format(os.path.dirname(path)))

        with open(path, "w") as file:
            for entity_id, sval_id in self.fake_entity2sval.items():
                line = "{},{}\n".format(entity_id, sval_id)
                file.write(line)


        # save_index(path, self.fake_entity2id.values(), self.fake_entity2id)

    @staticmethod
    def from_file(fake_entities_index_path, entity_index_path):
        manager = FakeEntityManager()

        logger.debug('loading real entities from: {}'.format(entity_index_path))
        manager.entity2id = read_index_file([entity_index_path])

        logger.debug('loading fake entities from: {}'.format(fake_entities_index_path))
        if fake_entities_index_path is not None:
            with open(fake_entities_index_path, "r") as file:
                for line in file:
                    entity_id, sval_id = list(map(int, line.rstrip().split(",")))
                    manager.fake_entity2sval[entity_id] = sval_id
                    # manager.fake_entity2id = read_index_file([fake_entities_index_path])
        manager.update_current_max_id()
        manager.update_fake_entity_ids()

        return manager

    def get_fake_entity_ids_in_entities(self, entity_ids):
        logger.debug("entity_ids: {}".format(entity_ids))
        logger.debug("fake_entity_ids: {}".format(self.fake_entity_ids))
        logger.debug("fake_entity2sval: {}".format(self.fake_entity2sval))
        return self.fake_entity_ids.intersection(set(entity_ids))

    def __str__(self):
        return str(self.fake_entity2sval)