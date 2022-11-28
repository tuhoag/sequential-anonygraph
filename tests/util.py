import os
import sys

def add_main_module_package():
    if __name__ == "__main__" and __package__ is None:
        sys.path.append(os.path.dirname(os.path[0]))
        # __package__ =