from numpy import *


def test():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    list1 = tile([0, 0], (4, 1))
    print(list1)
    diffMat = list1 - group
    print(diffMat)


test()
