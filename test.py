import math


def reu(array):
    mid = math.floor(len(array) / 2)
    left = array[0:mid]
    right = array[mid + 1:-1]
    left = reu(left)
    right = reu(right)
    print(left)
    print(right)


array = [1]
print(reu(array))
