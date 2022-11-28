import sys
from sortedcontainers import SortedList

class SortedValueDict():
    def __init__(self):
        # self.smallest_values =
        self.items = {}

    def __setitem__(self, key, value):
        self.items[key] = value
        # self.smallest_value = min(value, self.)

    def __delitem__(self, key):
        del self.items[key]



    def __len__(self):
        pass




