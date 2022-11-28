from sortedcontainers import SortedSet

class HistoricalMatrix:
    def __init__(self):
        self.__data = {}
        self.__time_insts = SortedSet()

    @property
    def entity_ids(self):
        return set(self.__data.keys())


    def add_new_entity_info(self, entity_id, attr_info, out_info, in_info, signature, time_inst):
        history_entry = self.__data.get(entity_id)
        if history_entry is None:
            history_entry = {}
            self.__data[entity_id] = history_entry

        history_entry[time_inst] = {
            'out': out_info,
            'in': in_info,
            'attr': attr_info,
            'sign': signature,
        }

        self.__time_insts.add(time_inst)
