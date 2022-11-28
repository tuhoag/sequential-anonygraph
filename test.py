from sortedcontainers import SortedList

list = SortedList(key=lambda item: (item[0], item[1]))

list.add((0, 0.2))
list.add((0, 0.1))
list.add((1, 0.6))
list.add((1, 0.4))

print(list)