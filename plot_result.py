'''generate plot for paper
'''

import matplotlib.pyplot as plt
import numpy as np
import tabulate
from input_output import IOUtility


class PlotResults:
    '''plotting results'''

    @staticmethod
    def preliminary(data):
        pass


def test():
    logs = ['result/11.14/log']
    data = IOUtility.read_logs(logs)
    print(data)


if __name__ == '__main__':
    test()