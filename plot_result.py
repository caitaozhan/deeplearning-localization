'''generate plot for paper
'''

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tabulate
from collections import defaultdict
from input_output import IOUtility, Default

from typing import List

class PlotResults:
    '''plotting results'''

    plt.rcParams['font.size'] = 65
    plt.rcParams['lines.linewidth'] = 10
    plt.rcParams['lines.markersize'] = 15
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'

    METHOD  = ['deepmtl', 'deepmtl-yolo', 'deepmtl-simple']
    _LEGEND = ['DeepMTL', 'DeepMTL-yolo', 'DeepMTL-peak']
    LEGEND  = dict(zip(METHOD, _LEGEND))

    METHOD  = ['deepmtl', 'deepmtl-yolo', 'deepmtl-simple']
    _COLOR  = ['r',       'tab:cyan',     'tab:orange']
    COLOR   = dict(zip(METHOD, _COLOR))

    METRIC = ['miss', 'false']
    _HATCH = ['*', '']
    HATCH  = dict(zip(METRIC, _HATCH))

    @staticmethod
    def reduce_avg(vals):
        vals = [val for val in vals if val is not None]
        vals = [val for val in vals if np.isnan(val) == False]
        return round(np.mean(vals), 4)


    @staticmethod
    def error_cdf(data, sen_density, num_intruder, fignames: List):
        metric = 'error'
        table = defaultdict(list)
        for myinput, output_by_method in data:
            if myinput.sensor_density == sen_density and myinput.num_intruder == num_intruder:
                for method, output in output_by_method.items():
                    table[method].append(output.get_metric(metric))

        # the plot
        n_bins = 100
        method_n_bins = []
        for method, error_list in table.items():
            n, bins, _ = plt.hist(error_list, n_bins, density=True, histtype='step', cumulative=True, label=method)
            method_n_bins.append((method, n, bins))
        plt.close()
        plt.rcParams['font.size'] = 60
        fig, ax = plt.subplots(figsize=(28, 18))
        fig.subplots_adjust(left=0.15, right=0.96, top=0.96, bottom=0.15)
        method_n_bins[1], method_n_bins[2] = method_n_bins[2], method_n_bins[1]  # switch ...
        for method, n, bins in method_n_bins:
            ax.plot(bins[1:], n, label=PlotResults.LEGEND[method], color=PlotResults.COLOR[method])

        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Localization Error (m)', labelpad=40)
        ax.set_ylabel('Cumulative Distribution (CDF)', labelpad=40)
        ax.set_ylim([0, 1.003])
        ax.set_xlim([0, 1.2])
        ax.tick_params(axis='x', pad=15)
        ax.tick_params(axis='y', pad=15)
        ax.set_xticklabels([0, 2, 4, 6, 8, 10, 12])
        ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
        for figname in fignames:
            fig.savefig(figname)


    @staticmethod
    def our_error_missfalse_vary_numintru(data, sen_density, fignames: List):
        # step 1: prepare for data
        metric = 'error'
        methods = ['deepmtl', 'deepmtl-yolo', 'deepmtl-simple']
        table = defaultdict(list)
        reduce_f = PlotResults.reduce_avg
        for myinput, output_by_method in data:
            if myinput.sensor_density == sen_density and myinput.num_intruder in [1, 3, 5, 7, 10]:
                table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
        arr = np.array(print_table)
        deepmtl_error      = arr[:, 1] * Default.cell_length
        deepmtl_yolo_error = arr[:, 2] * Default.cell_length
        deepmtl_peak_error = arr[:, 3] * Default.cell_length

        metric = 'miss'
        table = defaultdict(list)
        for myinput, output_by_method in data:
            if myinput.sensor_density == Default.sen_density and myinput.num_intruder in [1, 3, 5, 7, 10]:
                num_tx = myinput.num_intruder
                table[num_tx].append({method: output.get_metric(metric)/num_tx for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
        arr = np.array(print_table)
        deepmtl_miss      = arr[:, 1] * 100
        deepmtl_yolo_miss = arr[:, 2] * 100
        deepmtl_peak_miss = arr[:, 3] * 100

        metric = 'false_alarm'
        table = defaultdict(list)
        for myinput, output_by_method in data:
            if myinput.sensor_density == Default.sen_density and myinput.num_intruder in [1, 3, 5, 7, 10]:
                num_tx = myinput.num_intruder
                table[num_tx].append({method: output.get_metric(metric)/output.get_pred_len() for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
        arr = np.array(print_table)
        deepmtl_false      = arr[:, 1] * 100
        deepmtl_yolo_false = arr[:, 2] * 100
        deepmtl_peak_false = arr[:, 3] * 100
        X_label            = arr[:, 0]

        # step 2: the plot
        plt.rcParams['font.size'] = 65
        ind = np.arange(len(deepmtl_error))
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(40, 20))
        fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.12)
        width = 0.24
        pos1 = ind - width - 0.005
        pos2 = ind
        pos3 = ind + width + 0.005
        ax0.bar(pos1, deepmtl_error, width, edgecolor='black', label=PlotResults.LEGEND['deepmtl'], color=PlotResults.COLOR['deepmtl'])
        ax0.bar(pos2, deepmtl_yolo_error, width, edgecolor='black', label=PlotResults.LEGEND['deepmtl-yolo'], color=PlotResults.COLOR['deepmtl-yolo'])
        ax0.bar(pos3, deepmtl_peak_error, width, edgecolor='black', label=PlotResults.LEGEND['deepmtl-simple'], color=PlotResults.COLOR['deepmtl-simple'])
        ax0.set_xticks(ind)
        ax0.set_xticklabels([str(int(x)) for x in X_label])
        ax0.tick_params(axis='x', pad=15)
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax0.set_ylabel('Mean Localization Error (m)')
        ax0.set_xlabel('Number of Intruders', labelpad=20)
        handles, labels = ax0.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=70)

        ax1.bar(pos1, deepmtl_miss, width, edgecolor='black',       color=PlotResults.COLOR['deepmtl'],        hatch=PlotResults.HATCH['miss'])
        ax1.bar(pos1, deepmtl_false, width, edgecolor='black',      color=PlotResults.COLOR['deepmtl'],        hatch=PlotResults.HATCH['false'], bottom=deepmtl_miss)
        ax1.bar(pos2, deepmtl_yolo_miss, width, edgecolor='black',  color=PlotResults.COLOR['deepmtl-yolo'],   hatch=PlotResults.HATCH['miss'])
        ax1.bar(pos2, deepmtl_yolo_false, width, edgecolor='black', color=PlotResults.COLOR['deepmtl-yolo'],   hatch=PlotResults.HATCH['false'], bottom=deepmtl_yolo_miss)
        ax1.bar(pos3, deepmtl_peak_miss, width, edgecolor='black',  color=PlotResults.COLOR['deepmtl-simple'], hatch=PlotResults.HATCH['miss'])
        ax1.bar(pos3, deepmtl_peak_false, width, edgecolor='black', color=PlotResults.COLOR['deepmtl-simple'], hatch=PlotResults.HATCH['false'], bottom=deepmtl_peak_miss)
        miss_patch = mpatches.Patch(facecolor='0.95', hatch=PlotResults.HATCH['miss'], label='Miss Rate')
        false_patch = mpatches.Patch(facecolor='0.95', hatch=PlotResults.HATCH['false'], label='False Alarm Rate')
        ax1.legend(handles=[miss_patch, false_patch], bbox_to_anchor=(0.32, 0.52, 0.5, 0.5))
        ax1.set_xticks(ind)
        ax1.set_xticklabels([str(int(x)) for x in X_label])
        ax1.tick_params(axis='x', pad=15)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax1.set_xlabel('Number of Intruders', labelpad=20)
        ax1.set_ylabel('Percentage (%)')

        for figname in fignames:
            fig.savefig(figname)


    @staticmethod
    def our_error_missfalse_vary_sendensity(data, num_intruder, fignames: List):
        # step 1: prepare for data
        metric = 'error'
        methods = ['deepmtl', 'deepmtl-yolo', 'deepmtl-simple']
        table = defaultdict(list)
        reduce_f = PlotResults.reduce_avg
        for myinput, output_by_method in data:
            if myinput.num_intruder == num_intruder:
                table[myinput.sensor_density].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['SEN DEN'] + methods))
        arr = np.array(print_table)
        deepmtl_error      = arr[:, 1] * Default.cell_length
        deepmtl_yolo_error = arr[:, 2] * Default.cell_length
        deepmtl_peak_error = arr[:, 3] * Default.cell_length

        metric = 'miss'
        table = defaultdict(list)
        for myinput, output_by_method in data:
            if myinput.num_intruder == num_intruder:
                num_tx = myinput.num_intruder
                table[myinput.sensor_density].append({method: output.get_metric(metric)/num_tx for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['SEN DEN'] + methods))
        arr = np.array(print_table)
        deepmtl_miss      = arr[:, 1] * 100
        deepmtl_yolo_miss = arr[:, 2] * 100
        deepmtl_peak_miss = arr[:, 3] * 100

        metric = 'false_alarm'
        table = defaultdict(list)
        for myinput, output_by_method in data:
            if myinput.num_intruder == num_intruder:
                table[myinput.sensor_density].append({method: output.get_metric(metric)/output.get_pred_len() for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['SEN DEN'] + methods))
        arr = np.array(print_table)
        deepmtl_false      = arr[:, 1] * 100
        deepmtl_yolo_false = arr[:, 2] * 100
        deepmtl_peak_false = arr[:, 3] * 100
        X_label            = arr[:, 0]

        # step 2: the plot
        plt.rcParams['font.size'] = 65
        ind = np.arange(len(deepmtl_error))
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(40, 20))
        fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.12)
        width = 0.24
        pos1 = ind - width - 0.005
        pos2 = ind
        pos3 = ind + width + 0.005
        ax0.bar(pos1, deepmtl_error, width, edgecolor='black', label=PlotResults.LEGEND['deepmtl'], color=PlotResults.COLOR['deepmtl'])
        ax0.bar(pos2, deepmtl_yolo_error, width, edgecolor='black', label=PlotResults.LEGEND['deepmtl-yolo'], color=PlotResults.COLOR['deepmtl-yolo'])
        ax0.bar(pos3, deepmtl_peak_error, width, edgecolor='black', label=PlotResults.LEGEND['deepmtl-simple'], color=PlotResults.COLOR['deepmtl-simple'])
        ax0.set_xticks(ind)
        ax0.set_xticklabels([str(int(int(x)/10000*100)) for x in X_label])
        ax0.tick_params(axis='x', pad=15)
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax0.set_ylabel('Mean Localization Error (m)')
        ax0.set_xlabel('Sensor Density (%)', labelpad=20)
        handles, labels = ax0.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=70)

        ax1.bar(pos1, deepmtl_miss, width, edgecolor='black',       color=PlotResults.COLOR['deepmtl'],        hatch=PlotResults.HATCH['miss'])
        ax1.bar(pos1, deepmtl_false, width, edgecolor='black',      color=PlotResults.COLOR['deepmtl'],        hatch=PlotResults.HATCH['false'], bottom=deepmtl_miss)
        ax1.bar(pos2, deepmtl_yolo_miss, width, edgecolor='black',  color=PlotResults.COLOR['deepmtl-yolo'],   hatch=PlotResults.HATCH['miss'])
        ax1.bar(pos2, deepmtl_yolo_false, width, edgecolor='black', color=PlotResults.COLOR['deepmtl-yolo'],   hatch=PlotResults.HATCH['false'], bottom=deepmtl_yolo_miss)
        ax1.bar(pos3, deepmtl_peak_miss, width, edgecolor='black',  color=PlotResults.COLOR['deepmtl-simple'], hatch=PlotResults.HATCH['miss'])
        ax1.bar(pos3, deepmtl_peak_false, width, edgecolor='black', color=PlotResults.COLOR['deepmtl-simple'], hatch=PlotResults.HATCH['false'], bottom=deepmtl_peak_miss)
        miss_patch = mpatches.Patch(facecolor='0.95', hatch=PlotResults.HATCH['miss'], label='Miss Rate')
        false_patch = mpatches.Patch(facecolor='0.95', hatch=PlotResults.HATCH['false'], label='False Alarm Rate')
        ax1.legend(handles=[miss_patch, false_patch], bbox_to_anchor=(0.2, 0.52, 0.5, 0.5))
        ax1.set_xticks(ind)
        ax1.set_xticklabels([str(int(int(x)/10000*100)) for x in X_label])
        ax1.tick_params(axis='x', pad=15)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax1.set_ylim([0, 20])
        ax1.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        ax1.set_xlabel('Sensor Density (%)', labelpad=20)
        ax1.set_ylabel('Percentage (%)')

        for figname in fignames:
            fig.savefig(figname)


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
    '''
        for error, do a cdf             (fix sensor density at 600, fix number of TX at 5)
        for miss and false, do a bar
        fix sensor density
    '''
    # error cdf plot
    # logs = ['result/12.9/log-deepmtl', 'result/12.9/log-deepmtl-simple2', 'result/12.10/log-deepmtl-yolo', 'result/12.10/log-deepmtl-all']
    # methods = ['deepmtl', 'deepmtl-simple', 'deepmtl-yolo']
    # data = IOUtility.read_logs(logs)
    # fignames = ['result/12.10/ours_cdf.png', '/home/caitao/Project/latex/localize/deeplearning/figures/ours_cdf.png']
    # PlotResults.error_cdf(data, sen_density=600, num_intruder=1, fignames=fignames)

    # # error, miss and false varying number of intruder
    # logs = ['result/12.10/log-deepmtl-all-vary_numintru']
    # methods = ['deepmtl', 'deepmtl-simple', 'deepmtl-yolo']
    # data = IOUtility.read_logs(logs)
    # fignames = ['result/12.10/ours_error_missfalse_vary_numintru.png', '/home/caitao/Project/latex/localize/deeplearning/figures/ours_error_missfalse_vary_numintru.png']
    # PlotResults.our_error_missfalse_vary_numintru(data, sen_density=Default.sen_density, fignames=fignames)

    # error, miss and false varying sensor density
    # logs = ['result/12.10/log-deepmtl-all-vary_sendensity']
    # methods = ['deepmtl', 'deepmtl-simple', 'deepmtl-yolo']
    # data = IOUtility.read_logs(logs)
    # fignames = ['result/12.10/ours_error_missfalse_vary_sendensity.png', '/home/caitao/Project/latex/localize/deeplearning/figures/ours_error_missfalse_vary_sendensity.png']
    # PlotResults.our_error_missfalse_vary_sendensity(data, num_intruder=Default.num_intruder, fignames=fignames)

    # time
    logs = ['result/12.10/log-deepmtl-all-time']
    methods = ['deepmtl', 'deepmtl-simple', 'deepmtl-yolo']
    data = IOUtility.read_logs(logs)
    metric = 'time'
    table = defaultdict(list)
    reduce_f = PlotResults.reduce_avg
    for myinput, output_by_method in data:
        if myinput.sensor_density == Default.sen_density and myinput:
            table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
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
