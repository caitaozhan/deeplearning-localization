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

    METHOD  = ['deepmtl', 'deepmtl-yolo', 'deepmtl-simple', 'map',     'splot', 'dtxf',         'deepmtl_noretrain',          'predpower',                   'predpower_nocorrect',       'deepmtl_auth',                       'deepmtl_auth_subtractpower']
    _LEGEND = ['DeepMTL', 'DeepMTL-yolo', 'DeepMTL-peak',   'MAP$^*$', 'SPLOT', 'DeepTxFinder', 'DeepMTL(No Step 2 Retrain)', 'PredPower (With Correction)', 'PredPower (No Correction)', 'Localize, Remove Authorized TX', 'Subtract Authorized TX Power, Localize']
    LEGEND  = dict(zip(METHOD, _LEGEND))

    METHOD  = ['deepmtl', 'deepmtl-yolo', 'deepmtl-simple', 'map',       'splot',       'dtxf', 'deepmtl_noretrain', 'predpower', 'predpower_nocorrect', 'deepmtl_auth', 'deepmtl_auth_subtractpower']
    _COLOR  = ['r',       'tab:cyan',     'tab:orange',     'limegreen', 'deepskyblue', 'gold', 'tab:purple',        'tab:gray',  'lightgray',           'fuchsia',      'lightpink']
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
        fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.16)
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
        ax0.set_xlabel('Number of Transmitters', labelpad=10)
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
        ax1.legend(handles=[false_patch, miss_patch], bbox_to_anchor=(0.32, 0.52, 0.5, 0.5))
        ax1.set_xticks(ind)
        ax1.set_xticklabels([str(int(x)) for x in X_label])
        ax1.tick_params(axis='x', pad=15)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax1.set_xlabel('Number of Transmitters', labelpad=10)
        ax1.set_ylabel('Percentage (%)')
        plt.figtext(0.265, 0.01, '(a)',  weight='bold')
        plt.figtext(0.757, 0.01, '(b)',  weight='bold')
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
        fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.16)
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
        ax0.set_xlabel('Sensor Density (%)', labelpad=10)
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
        ax1.legend(handles=[false_patch, miss_patch], bbox_to_anchor=(0.2, 0.52, 0.5, 0.5))
        ax1.set_xticks(ind)
        ax1.set_xticklabels([str(int(int(x)/10000*100)) for x in X_label])
        ax1.tick_params(axis='x', pad=15)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax1.set_ylim([0, 20])
        ax1.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        ax1.set_xlabel('Sensor Density (%)', labelpad=10)
        ax1.set_ylabel('Percentage (%)')
        plt.figtext(0.265, 0.01, '(a)',  weight='bold')
        plt.figtext(0.757, 0.01, '(b)',  weight='bold')
        for figname in fignames:
            fig.savefig(figname)


    @staticmethod
    def error_missfalse_vary_numintru(data, data_source, sen_density, fignames: List):
        # step 1: prepare for data
        metric = 'error'
        methods = ['deepmtl', 'map', 'splot', 'deeptxfinder']
        table = defaultdict(list)
        reduce_f = PlotResults.reduce_avg
        for myinput, output_by_method in data:
            if myinput.sensor_density == sen_density and myinput.data_source == data_source:
                table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
        arr = np.array(print_table)
        deepmtl_error = arr[:, 1] * Default.cell_length
        map_error     = arr[:, 2] * Default.cell_length
        splot_error   = arr[:, 3] * Default.cell_length
        dtxf_error    = arr[:, 4] * Default.cell_length

        metric = 'miss'
        table = defaultdict(list)
        for myinput, output_by_method in data:
            if myinput.sensor_density == Default.sen_density and myinput.data_source == data_source:
                num_tx = myinput.num_intruder
                table[num_tx].append({method: output.get_metric(metric)/num_tx for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
        arr = np.array(print_table)
        deepmtl_miss = arr[:, 1] * 100
        map_miss     = arr[:, 2] * 100
        splot_miss   = arr[:, 3] * 100
        dtxf_miss    = arr[:, 4] * 100

        metric = 'false_alarm'
        table = defaultdict(list)
        for myinput, output_by_method in data:
            if myinput.sensor_density == Default.sen_density and myinput.data_source == data_source:
                num_tx = myinput.num_intruder
                table[num_tx].append({method: output.get_metric(metric)/output.get_pred_len() for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
        arr = np.array(print_table)
        deepmtl_false = arr[:, 1] * 100
        map_false     = arr[:, 2] * 100
        splot_false   = arr[:, 3] * 100
        dtxf_false    = arr[:, 4] * 100
        X_label       = arr[:, 0]

        # step 2: the plot
        plt.rcParams['font.size'] = 65
        ind = np.arange(len(deepmtl_error))
        fig, ax = plt.subplots(1, 1, figsize=(40, 20))
        fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.12)
        width = 0.2
        pos1 = ind - 1.5 * width
        pos2 = ind - 0.5 * width
        pos3 = ind + 0.5 * width
        pos4 = ind + 1.5 * width
        ax.bar(pos1, deepmtl_error, width, edgecolor='black', label=PlotResults.LEGEND['deepmtl'], color=PlotResults.COLOR['deepmtl'])
        ax.bar(pos2, map_error, width, edgecolor='black', label=PlotResults.LEGEND['map'], color=PlotResults.COLOR['map'])
        ax.bar(pos3, splot_error, width, edgecolor='black', label=PlotResults.LEGEND['splot'], color=PlotResults.COLOR['splot'])
        ax.bar(pos4, dtxf_error, width, edgecolor='black', label=PlotResults.LEGEND['dtxf'], color=PlotResults.COLOR['dtxf'])
        ax.set_xticks(ind)
        ax.set_xticklabels([str(int(x)) for x in X_label])
        ax.tick_params(axis='x', pad=15)
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax.set_ylabel('Mean Localization Error (m)')
        ax.set_xlabel('Number of Transmitters', labelpad=20)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.064, 0.85), loc='lower left', ncol=4, fontsize=70)
        for figname in fignames[:2]:
            fig.savefig(figname)

        plt.rcParams['font.size'] = 70
        ind = np.arange(len(deepmtl_error))
        fig, ax = plt.subplots(1, 1, figsize=(40, 20))
        fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.12)
        ax.bar(pos1, deepmtl_miss, width, edgecolor='black',  color=PlotResults.COLOR['deepmtl'], hatch=PlotResults.HATCH['miss'])
        ax.bar(pos1, deepmtl_false, width, edgecolor='black', color=PlotResults.COLOR['deepmtl'], hatch=PlotResults.HATCH['false'], bottom=deepmtl_miss)
        ax.bar(pos2, map_miss, width, edgecolor='black',      color=PlotResults.COLOR['map'],     hatch=PlotResults.HATCH['miss'])
        ax.bar(pos2, map_false, width, edgecolor='black',     color=PlotResults.COLOR['map'],     hatch=PlotResults.HATCH['false'], bottom=map_miss)
        ax.bar(pos3, splot_miss, width, edgecolor='black',    color=PlotResults.COLOR['splot'],   hatch=PlotResults.HATCH['miss'])
        ax.bar(pos3, splot_false, width, edgecolor='black',   color=PlotResults.COLOR['splot'],   hatch=PlotResults.HATCH['false'], bottom=splot_miss)
        ax.bar(pos4, dtxf_miss, width, edgecolor='black',     color=PlotResults.COLOR['dtxf'],    hatch=PlotResults.HATCH['miss'])
        ax.bar(pos4, dtxf_false, width, edgecolor='black',    color=PlotResults.COLOR['dtxf'],    hatch=PlotResults.HATCH['false'], bottom=dtxf_miss)

        miss_patch = mpatches.Patch(facecolor='0.95', hatch=PlotResults.HATCH['miss'], label='Miss Rate')
        false_patch = mpatches.Patch(facecolor='0.95', hatch=PlotResults.HATCH['false'], label='False Alarm Rate')
        first_legend = plt.legend(handles=[false_patch, miss_patch], bbox_to_anchor=(0.02, 0.6, 0.4, 0.4))
        plt.gca().add_artist(first_legend)

        deepmtl_patch = mpatches.Patch(label=PlotResults.LEGEND['deepmtl'], color=PlotResults.COLOR['deepmtl'])
        map_patch = mpatches.Patch(label=PlotResults.LEGEND['map'], color=PlotResults.COLOR['map'])
        splot_patch = mpatches.Patch(label=PlotResults.LEGEND['splot'], color=PlotResults.COLOR['splot'])
        dtxf_patch = mpatches.Patch(label=PlotResults.LEGEND['dtxf'], color=PlotResults.COLOR['dtxf'])
        plt.legend(handles=[deepmtl_patch, map_patch, splot_patch, dtxf_patch], bbox_to_anchor=(0.52, 0.68, 0.5, 0.5), ncol=4)

        ax.set_xticks(ind)
        ax.set_xticklabels([str(int(x)) for x in X_label])
        ax.tick_params(axis='x', pad=15)
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax.set_xlabel('Number of Transmitters', labelpad=20)
        ax.set_ylabel('Percentage (%)')
        if data_source == 'data/305test':
            ax.set_ylim([0, 15])
            ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
        for figname in fignames[2:]:
            fig.savefig(figname)

    @staticmethod
    def noretrain_error_missfalse_vary_numintru(data, data_source, fignames):
        # step 1: prepare the data
        metric = 'error'
        methods = ['deepmtl', 'deepmtl_replacepart2']
        table = defaultdict(list)
        reduce_f = PlotResults.reduce_avg
        for myinput, output_by_method in data:
            if myinput.data_source == data_source:
                table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
        arr = np.array(print_table)
        deepmtl_error = arr[:, 1] * Default.cell_length
        deepmtl_noretrain_error = arr[:, 2] * Default.cell_length

        metric = 'miss'
        table = defaultdict(list)
        for myinput, output_by_method in data:
            if myinput.data_source == data_source:
                num_tx = myinput.num_intruder
                table[num_tx].append({method: output.get_metric(metric)/num_tx for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table,headers=['NUM TX'] + methods))
        arr = np.array(print_table)
        deepmtl_miss = arr[:, 1] * 100
        deepmtl_noretrain_miss = arr[:, 2] * 100

        metric = 'false_alarm'
        table = defaultdict(list)
        for myinput, output_by_method in data:
            if myinput.data_source == data_source:
                num_tx = myinput.num_intruder
                table[num_tx].append({method: output.get_metric(metric)/output.get_pred_len() for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
        arr = np.array(print_table)
        deepmtl_false = arr[:, 1] * 100
        deepmtl_noretrain_false = arr[:, 2] * 100
        X_label = arr[:, 0]

        # step 2: the plot
        plt.rcParams['font.size'] = 45
        ind = np.arange(len(deepmtl_error))
        fig, ax = plt.subplots(1, 1, figsize=(20, 13))
        fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.13)
        width = 0.35
        pos1 = ind - 0.5 * width
        pos2 = ind + 0.5 * width
        ax.bar(pos1, deepmtl_error, width, edgecolor='black', label=PlotResults.LEGEND['deepmtl'], color=PlotResults.COLOR['deepmtl'])
        ax.bar(pos2, deepmtl_noretrain_error, width, edgecolor='black', label=PlotResults.LEGEND['deepmtl_noretrain'], color=PlotResults.COLOR['deepmtl_noretrain'])
        ax.set_xticks(ind)
        ax.set_xticklabels([str(int(x)) for x in X_label])
        ax.tick_params(axis='x', pad=15)
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax.set_ylim([0, 6.9])
        ax.set_ylabel('Mean Localization Error (m)')
        ax.set_xlabel('Number of Transmitters', labelpad=20)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=45)
        fig.savefig(fignames[0])

        plt.rcParams['font.size'] = 70
        ind = np.arange(len(deepmtl_error))
        fig, ax = plt.subplots(1, 1, figsize=(40, 20))
        fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.12)
        ax.bar(pos1, deepmtl_miss, width, edgecolor='black', color=PlotResults.COLOR['deepmtl'], hatch=PlotResults.HATCH['miss'])
        ax.bar(pos1, deepmtl_false, width, edgecolor='black', color=PlotResults.COLOR['deepmtl'], hatch=PlotResults.HATCH['false'], bottom=deepmtl_miss)
        ax.bar(pos2, deepmtl_noretrain_miss, width, edgecolor='black',      color=PlotResults.COLOR['deepmtl_noretrain'],     hatch=PlotResults.HATCH['miss'])
        ax.bar(pos2, deepmtl_noretrain_false, width, edgecolor='black',     color=PlotResults.COLOR['deepmtl_noretrain'],     hatch=PlotResults.HATCH['false'], bottom=deepmtl_noretrain_miss)
        miss_patch = mpatches.Patch(facecolor='0.9', hatch=PlotResults.HATCH['miss'], label='Miss Rate')
        false_patch = mpatches.Patch(facecolor='0.9', hatch=PlotResults.HATCH['false'], label='False Alarm Rate')
        first_legend = plt.legend(handles=[false_patch, miss_patch], bbox_to_anchor=(0.02, 0.6, 0.4, 0.4))
        plt.gca().add_artist(first_legend)
        deepmtl_patch = mpatches.Patch(label=PlotResults.LEGEND['deepmtl'], color=PlotResults.COLOR['deepmtl'])
        deepmtl_noretrain_patch = mpatches.Patch(label=PlotResults.LEGEND['deepmtl_noretrain'], color=PlotResults.COLOR['deepmtl_noretrain'])
        plt.legend(handles=[deepmtl_patch, deepmtl_noretrain_patch], bbox_to_anchor=(0.4, 0.68, 0.5, 0.5), ncol=4)
        ax.set_xticks(ind)
        ax.set_xticklabels([str(int(x)) for x in X_label])
        ax.tick_params(axis='x', pad=15)
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax.set_xlabel('Number of Transmitters', labelpad=20)
        ax.set_ylabel('Percentage (%)')
        fig.savefig(fignames[1])


    @staticmethod
    def error_missfalse_vary_sendensity(data, data_source, num_intruder, fignames: List):
        # step 1: prepare for data
        metric = 'error'
        methods = ['deepmtl', 'map', 'splot', 'deeptxfinder']
        table = defaultdict(list)
        reduce_f = PlotResults.reduce_avg
        for myinput, output_by_method in data:
            if myinput.num_intruder == num_intruder and myinput.data_source == data_source:
                table[myinput.sensor_density].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['SEN DEN'] + methods))
        arr = np.array(print_table)
        deepmtl_error = arr[:, 1] * Default.cell_length
        map_error     = arr[:, 2] * Default.cell_length
        splot_error   = arr[:, 3] * Default.cell_length
        dtxf_error    = arr[:, 4] * Default.cell_length

        metric = 'miss'
        table = defaultdict(list)
        for myinput, output_by_method in data:
            if myinput.num_intruder == num_intruder and myinput.data_source == data_source:
                num_tx = myinput.num_intruder
                table[myinput.sensor_density].append({method: output.get_metric(metric)/num_tx for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['SEN DEN'] + methods))
        arr = np.array(print_table)
        deepmtl_miss = arr[:, 1] * 100
        map_miss     = arr[:, 2] * 100
        splot_miss   = arr[:, 3] * 100
        dtxf_miss    = arr[:, 4] * 100

        metric = 'false_alarm'
        table = defaultdict(list)
        for myinput, output_by_method in data:
            if myinput.num_intruder == num_intruder and myinput.data_source == data_source:
                table[myinput.sensor_density].append({method: output.get_metric(metric)/output.get_pred_len() for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['SEN DEN'] + methods))
        arr = np.array(print_table)
        deepmtl_false = arr[:, 1] * 100
        map_false     = arr[:, 2] * 100
        splot_false   = arr[:, 3] * 100
        dtxf_false    = arr[:, 4] * 100
        X_label       = arr[:, 0]

        # step 2: the plot
        plt.rcParams['font.size'] = 65
        ind = np.arange(len(deepmtl_error))
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(40, 20))
        fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.16)
        width = 0.18
        pos1 = ind - 1.5 * width
        pos2 = ind - 0.5 * width
        pos3 = ind + 0.5 * width
        pos4 = ind + 1.5 * width
        ax0.bar(pos1, deepmtl_error, width, edgecolor='black', label=PlotResults.LEGEND['deepmtl'], color=PlotResults.COLOR['deepmtl'])
        ax0.bar(pos2, map_error, width, edgecolor='black', label=PlotResults.LEGEND['map'], color=PlotResults.COLOR['map'])
        ax0.bar(pos3, splot_error, width, edgecolor='black', label=PlotResults.LEGEND['splot'], color=PlotResults.COLOR['splot'])
        ax0.bar(pos4, dtxf_error, width, edgecolor='black', label=PlotResults.LEGEND['dtxf'], color=PlotResults.COLOR['dtxf'])
        ax0.set_xticks(ind)
        ax0.set_xticklabels([str(int(int(x)/10000*100)) for x in X_label])
        ax0.tick_params(axis='x', pad=15)
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax0.set_ylabel('Mean Localization Error (m)')
        ax0.set_xlabel('Sensor Density (%)', labelpad=10)
        handles, labels = ax0.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=65)

        ax1.bar(pos1, deepmtl_miss, width, edgecolor='black',  color=PlotResults.COLOR['deepmtl'], hatch=PlotResults.HATCH['miss'])
        ax1.bar(pos1, deepmtl_false, width, edgecolor='black', color=PlotResults.COLOR['deepmtl'], hatch=PlotResults.HATCH['false'], bottom=deepmtl_miss)
        ax1.bar(pos2, map_miss, width, edgecolor='black',      color=PlotResults.COLOR['map'],     hatch=PlotResults.HATCH['miss'])
        ax1.bar(pos2, map_false, width, edgecolor='black',     color=PlotResults.COLOR['map'],     hatch=PlotResults.HATCH['false'], bottom=map_miss)
        ax1.bar(pos3, splot_miss, width, edgecolor='black',    color=PlotResults.COLOR['splot'],   hatch=PlotResults.HATCH['miss'])
        ax1.bar(pos3, splot_false, width, edgecolor='black',   color=PlotResults.COLOR['splot'],   hatch=PlotResults.HATCH['false'], bottom=splot_miss)
        ax1.bar(pos4, dtxf_miss, width, edgecolor='black',     color=PlotResults.COLOR['dtxf'],    hatch=PlotResults.HATCH['miss'])
        ax1.bar(pos4, dtxf_false, width, edgecolor='black',    color=PlotResults.COLOR['dtxf'],    hatch=PlotResults.HATCH['false'], bottom=dtxf_miss)
        miss_patch = mpatches.Patch(facecolor='0.95', hatch=PlotResults.HATCH['miss'], label='Miss Rate')
        false_patch = mpatches.Patch(facecolor='0.95', hatch=PlotResults.HATCH['false'], label='False Alarm Rate')
        ax1.legend(handles=[false_patch, miss_patch], bbox_to_anchor=(0.22, 0.5, 0.5, 0.5))
        ax1.set_xticks(ind)
        ax1.set_xticklabels([str(int(int(x)/10000*100)) for x in X_label])
        ax1.tick_params(axis='x', pad=15)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        # ax1.set_ylim([0, 14])
        # ax1.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax1.set_xlabel('Sensor Density (%)', labelpad=10)
        ax1.set_ylabel('Percentage (%)')
        plt.figtext(0.265, 0.01, '(a)',  weight='bold')
        plt.figtext(0.757, 0.01, '(b)',  weight='bold')
        if data_source == 'data/305test':
            ax1.set_ylim([0, 31])
        for figname in fignames:
            fig.savefig(figname)


    @staticmethod
    def powererror_varysensor(data, data_ipsn, data_splat, figname):
        '''evaluate power prediction error, single TX, varying sensor density
        '''
        X = [200, 400, 600, 800, 1000]
        predpower_abserror_logdist = [0.3386, 0.2387, 0.2092, 0.1923, 0.1993]
        # predpower_error_logdist = [-0.0068, -0.0229, -0.0017, 0.0201, -0.0077]
        predpower_abserror_splat = [0.2154, 0.1551, 0.1402, 0.1289, 0.1384]
        # predpower_error_splat = [0.0105, 0.015, 0.0281, 0.0107, -0.0079]
        
        reduce_f = PlotResults.reduce_avg
        metric = 'power_error'
        methods = ['map']
        table_splat = defaultdict(list)
        table_logdist = defaultdict(list)
        for myinput, output_by_method in data:
            if myinput.data_source == data_splat and myinput.num_intruder == 1:
                sen_density = myinput.sensor_density
                table_splat[sen_density].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
            if myinput.data_source == data_ipsn and myinput.num_intruder == 1:
                sen_density = myinput.sensor_density
                table_logdist[sen_density].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
        
        print_table_splat = []
        for x, list_of_y_by_method in sorted(table_splat.items()):
            tmp_list = [reduce_f([y_by_method[method] for y_by_method in list_of_y_by_method]) for method in methods]
            print_table_splat.append([x] + tmp_list)
        print_table_logdist = []
        for x, list_of_y_by_method in sorted(table_logdist.items()):
            tmp_list = [reduce_f([y_by_method[method] for y_by_method in list_of_y_by_method]) for method in methods]
            print_table_logdist.append([x] + tmp_list)

        print('Metric', metric)
        print(tabulate.tabulate(print_table_splat, headers=['NUM TX'] + methods))
        print(tabulate.tabulate(print_table_logdist, headers=['NUM TX'] + methods))

        arr = np.array(print_table_logdist)
        ipsn_abserror_logdist = arr[:, 1]
        arr = np.array(print_table_splat)
        ipsn_abserror_splat = arr[:, 1]
        X_label = arr[:, 0]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(40, 20))
        fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.16, wspace=0.25)
        ind = np.arange(len(X))
        width = 0.25
        pos1 = ind - 0.5*width - 0.005
        pos2 = ind + 0.5*width + 0.005
        ax0.bar(pos1, predpower_abserror_logdist, width, edgecolor='black', color=PlotResults.COLOR['predpower'], label=PlotResults.LEGEND['predpower'])
        ax0.bar(pos2, ipsn_abserror_logdist,      width, edgecolor='black', color=PlotResults.COLOR['map'],       label=PlotResults.LEGEND['map']      )
        ax0.legend(fontsize=45)
        ax0.set_ylim([0, 0.5])
        ax0.set_xticks(ind)
        ax0.set_xticklabels([str(int(int(x)/10000*100)) for x in X_label])
        ax0.tick_params(axis='x', pad=15)
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax0.set_ylabel('Power Estimation Error (dBm)', fontsize=55)
        ax0.set_xlabel('Sensor Density (%)', labelpad=10)
        ax0.set_title('Log Distance Model', fontsize=60, pad=25, weight='bold')

        ax1.bar(pos1, predpower_abserror_splat,   width, edgecolor='black', color=PlotResults.COLOR['predpower'], label=PlotResults.LEGEND['predpower'])
        ax1.bar(pos2, ipsn_abserror_splat,        width, edgecolor='black', color=PlotResults.COLOR['map'],       label=PlotResults.LEGEND['map']      )
        ax1.legend(fontsize=45)
        ax1.set_ylim([0, 0.5])
        ax1.set_xticks(ind)
        ax1.set_xticklabels([str(int(int(x)/10000*100)) for x in X_label])
        ax1.tick_params(axis='x', pad=15)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax1.set_ylabel('Power Estimation Error (dBm)', fontsize=55)
        ax1.set_xlabel('Sensor Density (%)', labelpad=10)
        ax1.set_title(' Longley-Rice Irregular Terrain \nWith Obstruction Model (SPLAT!)', fontsize=60, pad=25, weight='bold')
        plt.figtext(0.265, 0.01, '(a)',  weight='bold')
        plt.figtext(0.757, 0.01, '(b)',  weight='bold')

        fig.savefig(figname)


    @staticmethod
    def powererror_varyintru(data, sen_density, data_source, figname):
        
        # metric = 'error'
        # methods = ['map', 'predpower']
        # table = defaultdict(list)
        # reduce_f = PlotResults.reduce_avg
        # for myinput, output_by_method in data:
        #     if myinput.sensor_density == sen_density and myinput.data_source == data_source:
        #         table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
        # print_table = []
        # for x, list_of_y_by_method in sorted(table.items()):
        #     tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
        #     print_table.append([x] + tmp_list)
        # print('Metric:', metric)
        # print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))

        # metric = 'miss'
        # methods = ['map', 'predpower']
        # table = defaultdict(list)
        # reduce_f = PlotResults.reduce_avg
        # for myinput, output_by_method in data:
        #     if myinput.sensor_density == sen_density and myinput.data_source == data_source:
        #         table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
        # print_table = []
        # for x, list_of_y_by_method in sorted(table.items()):
        #     tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
        #     print_table.append([x] + tmp_list)
        # print('Metric', metric)
        # print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))

        # metric = 'false_alarm'
        # methods = ['map', 'predpower']
        # table = defaultdict(list)
        # reduce_f = PlotResults.reduce_avg
        # for myinput, output_by_method in data:
        #     if myinput.sensor_density == sen_density and myinput.data_source == data_source:
        #         table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
        # print_table = []
        # for x, list_of_y_by_method in sorted(table.items()):
        #     tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
        #     print_table.append([x] + tmp_list)
        # print('Metric:', metric)
        # print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))

        metric = 'power_error'
        methods = ['map', 'predpower_nocorrect', 'predpower']
        table = defaultdict(list)
        reduce_f = PlotResults.reduce_avg
        for myinput, output_by_method in data:
            if myinput.sensor_density == sen_density and myinput.data_source == data_source:
                table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
        arr = np.array(print_table)
        X_label = arr[:, 0]
        map = arr[:, 1]
        predpower_nocorrect = arr[:, 2]
        predpower = arr[:, 3]
        
        # the plot
        fig, ax = plt.subplots(1, 1, figsize=(40, 20))
        fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.12)
        ind = np.arange(len(map))
        width = 0.25
        pos1 = ind - width
        pos2 = ind
        pos3 = ind + width
        ax.bar(pos1, map, width, edgecolor='black', label=PlotResults.LEGEND['map'], color=PlotResults.COLOR['map'])
        ax.bar(pos2, predpower_nocorrect, width, edgecolor='black', label=PlotResults.LEGEND['predpower_nocorrect'], color=PlotResults.COLOR['predpower_nocorrect'])
        ax.bar(pos3, predpower, width, edgecolor='black', label=PlotResults.LEGEND['predpower'], color=PlotResults.COLOR['predpower'])
        ax.legend()
        ax.set_xticks(ind)
        ax.set_xticklabels([str(int(x)) for x in X_label])
        ax.tick_params(axis='x', pad=15)
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax.set_ylabel('Power Estimation Error (dBm)', fontsize=55)
        ax.set_xlabel('Number of Transmitters', labelpad=20)
        fig.savefig(figname)

    @staticmethod
    def error_authorized_varyintru(data, data_source, sen_density, fignames):
        # step 1: prepare for data
        metric = 'error'
        methods = ['deepmtl_auth','deepmtl_auth_subtractpower']
        table = defaultdict(list)
        reduce_f = PlotResults.reduce_avg
        for myinput, output_by_method in data:
            if myinput.sensor_density == sen_density and myinput.data_source == data_source:
                table[myinput.num_intruder].append({method:output.get_metric(metric) for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
        arr = np.array(print_table)
        deepmtl_auth_error               = arr[:, 1] * Default.cell_length
        deepmtl_auth_subtracepower_error = arr[:, 2] * Default.cell_length

        metric = 'miss'
        methods = ['deepmtl_auth', 'deepmtl_auth_subtractpower']
        table = defaultdict(list)
        reduce_f = PlotResults.reduce_avg
        for myinput, output_by_method in data:
            if myinput.sensor_density == sen_density and myinput.data_source == data_source:
                num_tx = myinput.num_intruder
                table[myinput.num_intruder].append({method:output.get_metric(metric)/num_tx for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
        arr = np.array(print_table)
        deepmtl_auth_miss               = arr[:, 1] * 100
        deepmtl_auth_subtractpower_miss = arr[:, 2] * 100
        X_label                         = arr[:, 0]

        metric = 'false_alarm'
        methods = ['deepmtl_auth', 'deepmtl_auth_subtractpower']
        table = defaultdict(list)
        reduce_f = PlotResults.reduce_avg
        for myinput, output_by_method in data:
            if myinput.sensor_density == sen_density and myinput.data_source == data_source:
                num_tx = myinput.num_intruder
                table[myinput.num_intruder].append({method: output.get_metric(metric)/num_tx for method, output in output_by_method.items()})
        print_table = []
        for x, list_of_y_by_method in sorted(table.items()):
            tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
            print_table.append([x] + tmp_list)
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
        arr = np.array(print_table)
        deepmtl_auth_false               = arr[:, 1] * 100
        deepmtl_auth_subtractpower_false = arr[:, 2] * 100

        # step 2: the plot
        plt.rcParams['font.size'] = 65
        ind = np.arange(len(deepmtl_auth_error))
        fig, ax = plt.subplots(1, 1, figsize=(40, 20))
        fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.12)
        width = 0.25
        pos1 = ind - 0.5*width
        pos2 = ind + 0.5*width
        ax.bar(pos1, deepmtl_auth_error, width, edgecolor='black', label=PlotResults.LEGEND['deepmtl_auth'], color=PlotResults.COLOR['deepmtl_auth'])
        ax.bar(pos2, deepmtl_auth_subtracepower_error, width, edgecolor='black', label=PlotResults.LEGEND['deepmtl_auth_subtractpower'], color=PlotResults.COLOR['deepmtl_auth_subtractpower'])
        ax.set_xticks(ind)
        ax.set_xticklabels([str(int(x)) for x in X_label])
        ax.tick_params(axis='x', pad=15)
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax.set_ylabel('Mean Localization Error (m)')
        ax.set_xlabel('Number of Intruders', labelpad=20)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.055, 0.88), loc='lower left', ncol=2, fontsize=55)
        fig.savefig(fignames[0])

        ind = np.arange(len(deepmtl_auth_error))
        fig, ax = plt.subplots(1, 1, figsize=(40, 20))
        fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.12)
        ax.bar(pos1, deepmtl_auth_miss, width, edgecolor='black',  color=PlotResults.COLOR['deepmtl_auth'], hatch=PlotResults.HATCH['miss'])
        ax.bar(pos1, deepmtl_auth_false, width, edgecolor='black', color=PlotResults.COLOR['deepmtl_auth'], hatch=PlotResults.HATCH['false'], bottom=deepmtl_auth_miss)
        ax.bar(pos2, deepmtl_auth_subtractpower_miss, width, edgecolor='black',  color=PlotResults.COLOR['deepmtl_auth_subtractpower'], hatch=PlotResults.HATCH['miss'])
        ax.bar(pos2, deepmtl_auth_subtractpower_false, width, edgecolor='black', color=PlotResults.COLOR['deepmtl_auth_subtractpower'], hatch=PlotResults.HATCH['false'], bottom=deepmtl_auth_subtractpower_miss)

        miss_patch = mpatches.Patch(facecolor='0.95', hatch=PlotResults.HATCH['miss'], label='Miss Rate')
        false_patch = mpatches.Patch(facecolor='0.95', hatch=PlotResults.HATCH['false'], label='False Alarm Rate')
        first_legend = plt.legend(handles=[false_patch, miss_patch], bbox_to_anchor=(0.15, 0.6, 0.4, 0.4), fontsize=55)
        plt.gca().add_artist(first_legend)

        deepmtl_auth_patch = mpatches.Patch(label=PlotResults.LEGEND['deepmtl_auth'], color=PlotResults.COLOR['deepmtl_auth'])
        deepmtl_auth_subtractpower_patch = mpatches.Patch(label=PlotResults.LEGEND['deepmtl_auth_subtractpower'], color=PlotResults.COLOR['deepmtl_auth_subtractpower'])
        plt.legend(handles=[deepmtl_auth_patch, deepmtl_auth_subtractpower_patch], bbox_to_anchor=(0.51, 0.68, 0.5, 0.5), ncol=2, fontsize=55)

        ax.set_xticks(ind)
        ax.set_xticklabels([str(int(x)) for x in X_label])
        ax.tick_params(axis='x', pad=15)
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax.set_xlabel('Number of Intruders', labelpad=20)
        ax.set_ylabel('Percentage (%)')
        fig.savefig(fignames[1])


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
    logs = ['result/12.9/log-deepmtl', 'result/12.9/log-deepmtl-simple2', 'result/12.10/log-deepmtl-yolo', 'result/12.10/log-deepmtl-all']
    methods = ['deepmtl', 'deepmtl-simple', 'deepmtl-yolo']
    data = IOUtility.read_logs(logs)
    fignames = ['result/12.10/ours_cdf.png', '/home/caitao/Project/latex/localize/deeplearning/figures/ours_cdf.png']
    PlotResults.error_cdf(data, sen_density=600, num_intruder=1, fignames=fignames)
    print()

    # error, miss and false varying number of intruder
    # logs = ['result/12.10/log-deepmtl-all-vary_numintru']
    logs = ['result/12.10/log-deepmtl-yolo_simple-vary_numintru', 'result/12.13/logdistance-deepmtl-5000']
    methods = ['deepmtl', 'deepmtl-simple', 'deepmtl-yolo']
    data = IOUtility.read_logs(logs)
    fignames = ['result/12.10/ours_error_missfalse_vary_numintru.png', '/home/caitao/Project/latex/localize/deeplearning/figures/ours_error_missfalse_vary_numintru.png']
    PlotResults.our_error_missfalse_vary_numintru(data, sen_density=Default.sen_density, fignames=fignames)
    print()

    # error, miss and false varying sensor density
    # logs = ['result/12.10/log-deepmtl-all-vary_sendensity']
    logs = ['result/12.10/log-deepmtl-yolo_simple-vary_sendensity', 'result/12.13/logdistance-deepmtl-5001']
    methods = ['deepmtl', 'deepmtl-simple', 'deepmtl-yolo']
    data = IOUtility.read_logs(logs)
    fignames = ['result/12.10/ours_error_missfalse_vary_sendensity.png', '/home/caitao/Project/latex/localize/deeplearning/figures/ours_error_missfalse_vary_sendensity.png']
    PlotResults.our_error_missfalse_vary_sendensity(data, num_intruder=Default.num_intruder, fignames=fignames)
    print()

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


def compare_logdistance():
    '''compare 1 + 3 over the ipsn data set
    '''

    # error, miss and false for varying num of intruders
    data_source = 'data/205test'
    logs    = ['result/12.11/log-map-5000', 'result/12.11/log-splot-5003', 'result/12.10/log-deepmtl-yolo_simple-vary_numintru', 'result/12.13/logdistance-deepmtl-5000', 'result/12.12/log-dtxf-5000']
    data    = IOUtility.read_logs(logs)
    fignames = ['result/12.12/log_distance-error_vary_numintru.png',     '/home/caitao/Project/latex/localize/deeplearning/figures/log_distance-error_vary_numintru.png',\
                'result/12.12/log_distance-missfalse_vary_numintru.png', '/home/caitao/Project/latex/localize/deeplearning/figures/log_distance-missfalse_vary_numintru.png']
    PlotResults.error_missfalse_vary_numintru(data, data_source, sen_density=Default.sen_density, fignames=fignames)
    print()

    # error, miss and false for varying sensor density
    logs    = ['result/12.11/log-map-5001', 'result/12.11/log-splot-5004', 'result/12.10/log-deepmtl-yolo_simple-vary_sendensity', 'result/12.13/logdistance-deepmtl-5001', 'result/12.12/log-dtxf-5001']
    data    = IOUtility.read_logs(logs)
    fignames = ['result/12.12/log_distance-error_missfalse_vary_sendensity.png',     '/home/caitao/Project/latex/localize/deeplearning/figures/log_distance-error_missfalse_vary_sendensity.png']
    PlotResults.error_missfalse_vary_sendensity(data, data_source, num_intruder=Default.num_intruder, fignames=fignames)
    print()

    # time
    logs = ['result/12.11/log-map-5000', 'result/12.11/log-splot-5003', 'result/12.10/log-deepmtl-all-vary_numintru', 'result/12.12/log-dtxf-5000']
    methods = ['deepmtl', 'map', 'splot', 'deeptxfinder']
    data = IOUtility.read_logs(logs)
    metric = 'time'
    table = defaultdict(list)
    reduce_f = PlotResults.reduce_avg
    for myinput, output_by_method in data:
        if myinput.sensor_density == Default.sen_density and myinput.num_intruder in [1, 3, 5, 7, 10]:
            table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
    print_table = []
    for x, list_of_y_by_method in sorted(table.items()):
        tmp_list = [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods]
        print_table.append([x] + tmp_list)
    print('Metric:', metric)
    print(tabulate.tabulate(print_table, headers=['NUM TX'] + methods))
    print()


def compare_splat():
    # error, miss and false for varying num of intruders
    data_source = 'data/305test'
    logs    = ['result/12.13/splat-deepmtl-5000', 'result/12.13/splat-map-5000-2', 'result/12.13/splat-dtxf-5003', 'result/12.13/splat-splot-5003']
    data    = IOUtility.read_logs(logs)
    fignames = ['result/12.13/splat-error_vary_numintru.png',     '/home/caitao/Project/latex/localize/deeplearning/figures/splat-error_vary_numintru.png',\
                'result/12.13/splat-missfalse_vary_numintru.png', '/home/caitao/Project/latex/localize/deeplearning/figures/splat-missfalse_vary_numintru.png']
    PlotResults.error_missfalse_vary_numintru(data, data_source, sen_density=Default.sen_density, fignames=fignames)
    print()

    logs    = ['result/12.13/splat-deepmtl-5001', 'result/12.13/splat-map-5001-2', 'result/12.13/splat-dtxf-5004', 'result/12.13/splat-splot-5004']
    data    = IOUtility.read_logs(logs)
    fignames = ['result/12.13/splat-error_missfalse_vary_sendensity.png',  '/home/caitao/Project/latex/localize/deeplearning/figures/splat-error_missfalse_vary_sendensity.png']
    PlotResults.error_missfalse_vary_sendensity(data, data_source, num_intruder=Default.num_intruder, fignames=fignames)
    print()


def noretrain():
    # testing data is logdistance, part 1 CNN is trained on log distance pipeline, 
    data_source = 'data/205test'
    logs = ['result/9.20/logdistance-deepmtl-5000', 'result/9.20/logdistance-deepmtl-5000_plus']
    data = IOUtility.read_logs(logs)
    fignames = ['result/9.20.2/noretrain-logdistance-error-vary_numintru.png', 'result/9.20.2/noretrain-logdistance-missfalse-vary_numintru.png']
    PlotResults.noretrain_error_missfalse_vary_numintru(data, data_source, fignames)
    data_source = 'data/305test'
    logs = ['result/9.20/splat-deepmtl-5000', 'result/9.20/splat-deepmtl-5000_plus']
    data = IOUtility.read_logs(logs)
    fignames = ['result/9.20.2/noretrain-splat-error-vary_numintru.png', 'result/9.20.2/noretrain-splat-missfalse-vary_numintru.png']
    PlotResults.noretrain_error_missfalse_vary_numintru(data, data_source, fignames)


def power_varysensor():
    data_ipsn = 'data/605test'
    data_splat = 'data/705test'
    logs = ['result/9.25/splat-map-5005', 'result/9.26/logdistance-map-5005', 'result/9.26/logdistance-map-5006', 
      'result/9.26/logdistance-map-5007', 'result/9.26/logdistance-map-5008', 'result/9.26/logdistance-map-5009']
    data = IOUtility.read_logs(logs)
    figname = 'result/9.25/powererror_varysensor.png'
    PlotResults.powererror_varysensor(data, data_ipsn, data_splat, figname)


def power_varyintru():
    data_ipsn = 'data/805test'
    data_splat = 'data/905test'
    logs = ['result/10.2/logdistance-map-5000', 'result/10.2/logdistance-deepmtl.predpower-5000', 'result/10.2/logdistance-deepmtl.predpower_nocorrect-5000']
    data = IOUtility.read_logs(logs)
    figname = 'result/10.2/logdist-powererror_varyintru.png'
    sen_density = 600
    PlotResults.powererror_varyintru(data, sen_density, data_ipsn, figname)


    logs = ['result/10.3/logdistance-deepmtl.predpower_and_nocorrect-5000', 'result/10.3/splat-map-5000', 'result/10.3/splat-map-5005',
            'result/10.3/splat-map-5006', 'result/10.3/splat-map-5007', 'result/10.3/splat-map-5008', 'result/10.3/splat-map-5009']
    data = IOUtility.read_logs(logs)
    figname = 'result/10.3/splat-powererror_varyintru.png'
    sen_density = 600
    PlotResults.powererror_varyintru(data, sen_density, data_splat, figname)


def authorized_varyintru():
    data_splat = 'data/1005test'
    logs = ['result/10.6-nms=0.4/splat-deepmtl_auth-5000', 'result/10.7/splat-deepmtl_auth_subtractpower3-5000-conf=0.85,nms=0.4']
    data = IOUtility.read_logs(logs)
    fignames = ['result/10.7/splat-error-authorized-varyintru.png', 'result/10.7/splat-missfalse-authorized-varyintru.png']
    sen_density = 600
    PlotResults.error_authorized_varyintru(data, data_splat, sen_density, fignames)


if __name__ == '__main__':
    # test()

    # compare_ours()
    # print('*'*20)
    # compare_logdistance()
    # print('*'*20)
    # compare_splat()

    # noretrain()

    # power_varysensor()

    # power_varyintru()

    authorized_varyintru()


###################### compare_ours()  ###########################
# Metric: error
#   NUM TX    deepmtl    deepmtl-yolo    deepmtl-simple
# --------  ---------  --------------  ----------------
#        1     0.2313          0.2758            0.3284
#        3     0.2202          0.2631            0.3256
#        5     0.2241          0.2684            0.3329
#        7     0.2358          0.2807            0.3376
#       10     0.2513          0.2976            0.3655
# Metric: miss
#   NUM TX    deepmtl    deepmtl-yolo    deepmtl-simple
# --------  ---------  --------------  ----------------
#        1     0               0                 0
#        3     0.0023          0.009             0.0067
#        5     0.0038          0.0097            0.0095
#        7     0.0067          0.014             0.0124
#       10     0.0102          0.0175            0.0196
# Metric: false_alarm
#   NUM TX    deepmtl    deepmtl-yolo    deepmtl-simple
# --------  ---------  --------------  ----------------
#        1     0               0.0096            0
#        3     0.0017          0.0104            0.0015
#        5     0.0017          0.008             0.0048
#        7     0.002           0.0087            0.0084
#       10     0.0025          0.0062            0.0111

# Metric: error
#   SEN DEN    deepmtl    deepmtl-yolo    deepmtl-simple
# ---------  ---------  --------------  ----------------
#       200     0.5436          0.703             0.6781
#       400     0.2937          0.362             0.3978
#       600     0.2241          0.2684            0.3329
#       800     0.1997          0.2454            0.3123
#      1000     0.2052          0.2615            0.3113
# Metric: miss
#   SEN DEN    deepmtl    deepmtl-yolo    deepmtl-simple
# ---------  ---------  --------------  ----------------
#       200     0.0175          0.0909            0.0367
#       400     0.0041          0.0245            0.0123
#       600     0.0038          0.0097            0.0095
#       800     0.0044          0.007             0.0111
#      1000     0.0055          0.0087            0.012
# Metric: false_alarm
#   SEN DEN    deepmtl    deepmtl-yolo    deepmtl-simple
# ---------  ---------  --------------  ----------------
#       200     0.0342          0.0869            0.1513
#       400     0.006           0.0288            0.0286
#       600     0.0017          0.008             0.0048
#       800     0.0016          0.01              0.0018
#      1000     0.001           0.0104            0.0013

# Metric: time
#   NUM TX    deepmtl    deepmtl-simple    deepmtl-yolo
# --------  ---------  ----------------  --------------
#        1     0.018             0.0013          0.018
#        3     0.0186            0.0014          0.0183
#        5     0.0189            0.0016          0.0192
#        7     0.0194            0.0018          0.0196
#       10     0.0206            0.0023          0.0205


#########################################################################
###################### compare_logdistance()  ###########################


#  NUM TX    deepmtl     map    splot    deeptxfinder
# --------  ---------  ------  -------  --------------
#        1     0.2313  0.7918   2.2435          1.0887
#        2     0.2159  0.7818   1.8396          3.688
#        3     0.2202  0.7855   1.9316          6.5036
#        4     0.218   0.8327   2.0261          7.5817
#        5     0.2241  0.8857   1.9223          8.8115
#        6     0.2258  0.8773   2.0771          9.29
#        7     0.2358  0.8855   2.0525          9.6476
#        8     0.2362  0.8521   1.9828          9.7892
#        9     0.2476  0.9495   2.0206         10.0733
#       10     0.2513  0.9529   2.0531         10.6988
# Metric: miss
#   NUM TX    deepmtl     map    splot    deeptxfinder
# --------  ---------  ------  -------  --------------
#        1     0       0.0116   0               0
#        2     0.0013  0.019    0               0.0099
#        3     0.0023  0.0057   0.009           0.0224
#        4     0.0018  0.017    0.0088          0.0301
#        5     0.0038  0.0114   0.0082          0.0331
#        6     0.0048  0.0119   0.0091          0.0398
#        7     0.0067  0.0361   0.0125          0.0431
#        8     0.0071  0.0413   0.0234          0.05
#        9     0.0102  0.0389   0.0181          0.0542
#       10     0.0102  0.0408   0.0255          0.0557
# Metric: false_alarm
#   NUM TX    deepmtl     map    splot    deeptxfinder
# --------  ---------  ------  -------  --------------
#        1     0       0        0               0.0027
#        2     0.0013  0.0032   0.0128          0.0099
#        3     0.0017  0.015    0.009           0.0294
#        4     0.0017  0.0039   0.0106          0.0433
#        5     0.0017  0.0048   0.0153          0.0452
#        6     0.0023  0.0187   0.014           0.056
#        7     0.002   0.0153   0.0305          0.0594
#        8     0.0019  0.0141   0.0195          0.0612
#        9     0.0022  0.0123   0.0291          0.0558
#       10     0.0025  0.0205   0.0374          0.0258

# Metric: error
#   SEN DEN    deepmtl     map    splot    deeptxfinder
# ---------  ---------  ------  -------  --------------
#       200     0.5436  1.1763   3.0707          9.1225
#       400     0.2937  0.936    2.3142          8.8304
#       600     0.2241  0.8553   1.9223          8.8115
#       800     0.1997  0.8394   1.8036          8.6713
#      1000     0.2052  0.8342   1.6338          8.6398
# Metric: miss
#   SEN DEN    deepmtl     map    splot    deeptxfinder
# ---------  ---------  ------  -------  --------------
#       200     0.0175  0.0396   0.0337          0.0641
#       400     0.0041  0.0255   0.0213          0.0449
#       600     0.0038  0.0162   0.0082          0.0331
#       800     0.0044  0.023    0.0125          0.0461
#      1000     0.0055  0.0297   0.0079          0.0462
# Metric: false_alarm
#   SEN DEN    deepmtl     map    splot    deeptxfinder
# ---------  ---------  ------  -------  --------------
#       200     0.0342  0.0069   0.0406          0.0635
#       400     0.006   0.0206   0.042           0.051
#       600     0.0017  0.0067   0.0153          0.0452
#       800     0.0016  0.0074   0.0149          0.0334
#      1000     0.001   0.0017   0.0243          0.0327

# Metric: time
#   NUM TX    deepmtl      map    splot    deeptxfinder
# --------  ---------  -------  -------  --------------
#        1     0.0179   8.7868   1.5399          0.0015
#        3     0.0185  15.1141   1.7998          0.0016
#        5     0.019   19.2941   2.063           0.0017
#        7     0.0196  24.0822   2.3205          0.0019
#       10     0.0206  28.4841   2.7188          0.0022


###################################################################
###################### compare_splat()  ###########################


# Metric: error
#   NUM TX    deepmtl     map    splot    deeptxfinder
# --------  ---------  ------  -------  --------------
#        1     0.5074  0.3769   2.2175          1.6199
#        2     0.5168  0.4239   2.0518          4.4177
#        3     0.5192  0.5653   2.2135          6.3188
#        4     0.5363  0.6823   2.3347          8.6476
#        5     0.5442  0.6926   2.1084          9.2813
#        6     0.559   0.8464   2.19            9.9123
#        7     0.5794  0.8842   2.4617         10.218
#        8     0.5882  0.903    2.471          10.0968
#        9     0.613   1.0269   2.6109         10.6644
#       10     0.6155  1.2364   2.7351         10.9353
# Metric: miss
#   NUM TX    deepmtl     map    splot    deeptxfinder
# --------  ---------  ------  -------  --------------
#        1     0       0        0               0
#        2     0.0024  0        0.0104          0.0344
#        3     0.0053  0.0208   0.0209          0.038
#        4     0.0062  0.0067   0.0349          0.0539
#        5     0.0101  0.008    0.0318          0.0752
#        6     0.0131  0.0099   0.0436          0.0713
#        7     0.0155  0.0101   0.0464          0.0689
#        8     0.0184  0.0077   0.0415          0.0719
#        9     0.0229  0.0134   0.0466          0.067
#       10     0.0264  0.0104   0.0472          0.0814
# Metric: false_alarm
#   NUM TX    deepmtl     map    splot    deeptxfinder
# --------  ---------  ------  -------  --------------
#        1     0.007   0        0               0.016
#        2     0.004   0        0.0017          0.0404
#        3     0.0058  0        0.0086          0.0582
#        4     0.0083  0.035    0.0034          0.0401
#        5     0.0105  0.0326   0.0126          0.0573
#        6     0.0105  0.0451   0.0133          0.0588
#        7     0.0125  0.0644   0.0185          0.0632
#        8     0.0145  0.0664   0.0277          0.0709
#        9     0.0134  0.0804   0.0456          0.0485
#       10     0.0155  0.0855   0.0485          0.023

# Metric: error
#   SEN DEN    deepmtl     map    splot    deeptxfinder
# ---------  ---------  ------  -------  --------------
#       200     1.0943  2.4247   3.0569          9.3426
#       400     0.6327  1.1195   2.4764          9.3157
#       600     0.5442  0.6926   2.1221          9.2813
#       800     0.502   0.5134   2.0305          9.241
#      1000     0.4789  0.4991   1.8141          9.3175
# Metric: miss
#   SEN DEN    deepmtl     map    splot    deeptxfinder
# ---------  ---------  ------  -------  --------------
#       200     0.0768  0.0324   0.1229          0.3033
#       400     0.0174  0.0104   0.0495          0.1157
#       600     0.0101  0.008    0.0325          0.0752
#       800     0.0081  0.0022   0.0313          0.0855
#      1000     0.0042  0.0065   0.0289          0.0889
# Metric: false_alarm
#   SEN DEN    deepmtl     map    splot    deeptxfinder
# ---------  ---------  ------  -------  --------------
#       200     0.102   0.1822   0.005           0
#       400     0.0247  0.0715   0.0095          0.0401
#       600     0.0105  0.0326   0.0129          0.0573
#       800     0.0078  0.0154   0.0132          0.0466
#      1000     0.007   0.0081   0.0113          0.0488



#########################################################################
###################### power_varyintru()  ###########################

# Metric: power_error
#   NUM TX     map    predpower_nocorrect    predpower
# --------  ------  ---------------------  -----------
#        1  0.175                  0.2069       0.2069
#        2  0.3463                 0.3775       0.2808
#        3  0.4356                 0.5374       0.3399
#        4  0.5292                 0.6783       0.3737
#        5  0.574                  0.7979       0.4133
#        6  0.6462                 0.9381       0.468
#        7  0.6924                 1.0961       0.509
#        8  0.7369                 1.2011       0.5514
#        9  0.7829                 1.3298       0.5732
#       10  0.8035                 1.4271       0.6067
# Metric: power_error
#   NUM TX     map    predpower_nocorrect    predpower
# --------  ------  ---------------------  -----------
#        1  0.4362                 0.145        0.145
#        2  0.6244                 0.4786       0.3338
#        3  0.7075                 0.7663       0.4363
#        4  0.8727                 1.0129       0.5345
#        5  1.0072                 1.3557       0.6307
#        6  1.1175                 1.6306       0.7242
#        7  1.2473                 1.8408       0.7991
#        8  1.3258                 2.1672       0.909
#        9  1.3921                 2.3211       0.9726
#       10  1.4768                 2.5541       1.0551