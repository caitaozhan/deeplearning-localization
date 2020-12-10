'''generate plot for paper
'''

import matplotlib.pyplot as plt
import numpy as np
import tabulate
from collections import defaultdict
from input_output import IOUtility, Default


class PlotResults:
    '''plotting results'''

    @staticmethod
    def reduce_avg(vals):
        vals = [val for val in vals if val is not None]
        vals = [val for val in vals if np.isnan(val) == False]
        return round(np.mean(vals), 3)


    @staticmethod
    def preliminary(data):
        pass


def test():
    logs = ['result/11.19/log-differentsensor']
    methods = ['dl', 'dl2', 'map']
    logs = ['result/11.19/log-differentsensor', 'result/11.27/log-differentsensor-2']
    methods = ['dl', 'dl2', 'dl3', 'map']
    data = IOUtility.read_logs(logs)
    table = defaultdict(list)
    # error
    metric = 'error'
    reduce_f = PlotResults.reduce_avg
    for myinput, output_by_method in data:
        table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
    print_table = []
    for x, list_of_y_by_method in sorted(table.items()):
        tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
        print_table.append([x] + tmp_list)
    print('Metric:', metric)
    print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
    print()

    # miss and false
    metric = 'miss'
    table = defaultdict(list)
    for myinput, output_by_method in data:
        num_tx = myinput.num_intruder
        table[num_tx].append({method: output.get_metric(metric)/num_tx for method, output in output_by_method.items()})
    print_table = []
    for x, list_of_y_by_method in sorted(table.items()):
        tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
        print_table.append([x] + tmp_list)
    print('Metric:', metric)
    print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
    print()

    metric = 'false_alarm'
    table = defaultdict(list)
    for myinput, output_by_method in data:
        num_tx = myinput.num_intruder
        table[num_tx].append({method: output.get_metric(metric)/output.get_pred_len() for method, output in output_by_method.items()})
    print_table = []
    for x, list_of_y_by_method in sorted(table.items()):
        tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
        print_table.append([x] + tmp_list)
    print('Metric:', metric)
    print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
    print()

    metric = 'time'
    reduce_f = PlotResults.reduce_avg
    for myinput, output_by_method in data:
        table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
    print_table = []
    for x, list_of_y_by_method in sorted(table.items()):
        tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
        print_table.append([x] + tmp_list)
    print('Metric:', metric)
    print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
    print()


def compare_ours():

    # for error, do a cdf             (fix sensor density at 600, fix number of TX at 5)
    # for miss and false, do a bar
    # fix sensor density 

    logs = ['result/12.9/log-deepmtl', 'result/12.9/log-deepmtl-simple2']
    methods = ['deepmtl', 'deepmtl-simple']
    data = IOUtility.read_logs(logs)
    table = defaultdict(list)

    # error
    metric = 'error'
    reduce_f = PlotResults.reduce_avg
    for myinput, output_by_method in data:
        # if myinput.num_intruder == Default.num_intruder and myinput.sensor_density == Default.sen_density:
        if myinput.sensor_density == Default.sen_density:
            table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
    print_table = []
    for x, list_of_y_by_method in sorted(table.items()):
        tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
        print_table.append([x] + tmp_list)
    print('Metric:', metric)
    print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
    print()


    # miss and false
    metric = 'miss'
    table = defaultdict(list)
    for myinput, output_by_method in data:
        if myinput.sensor_density == Default.sen_density:
            num_tx = myinput.num_intruder
            table[num_tx].append({method: output.get_metric(metric)/num_tx for method, output in output_by_method.items()})
    print_table = []
    for x, list_of_y_by_method in sorted(table.items()):
        tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
        print_table.append([x] + tmp_list)
    print('Metric:', metric)
    print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
    print()

    metric = 'false_alarm'
    table = defaultdict(list)
    for myinput, output_by_method in data:
        if myinput.sensor_density == Default.sen_density:
            num_tx = myinput.num_intruder
            table[num_tx].append({method: output.get_metric(metric)/output.get_pred_len() for method, output in output_by_method.items()})
    print_table = []
    for x, list_of_y_by_method in sorted(table.items()):
        tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
        print_table.append([x] + tmp_list)
    print('Metric:', metric)
    print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
    print()


if __name__ == '__main__':
    # test()

    compare_ours()
