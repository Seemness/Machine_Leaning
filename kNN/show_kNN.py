import matplotlib
import matplotlib.pyplot as plt
import kNN
from numpy import *


def show_file2matrix():
    fig = plt.figure()
    returnMat, classLabelVector = kNN.file2matrix()
    ax = fig.add_subplot(111)
    ax.scatter(returnMat[:, 1], returnMat[:, 2], 15.0 * array(classLabelVector), 15.0 * array(classLabelVector))
    plt.show()


show_file2matrix()
