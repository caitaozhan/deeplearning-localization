'''The client side, send localization request to the server
'''

import os
import json
import argparse
import time
import random
from subprocess import PIPE, Popen
from input_output import Input
from input_output import Default
import mydnn_util


def get_index_from_log(log: str):
    '''for repeating experiments
    '''
    index = []
    with open(log, 'r') as f:
        for line in f:
            try:
                myinput = Input.from_json_str(line)
                index.append(myinput.image_index)
            except:
                pass
    return index


def get_sen_num(idx, root_dir='data/205test', sample_per_label=5):
    folder = int(idx / sample_per_label)
    folder = format(folder, '06d')
    json_name = str(idx % sample_per_label) + '.json'
    json_path = os.path.join(root_dir, folder, json_name)
    with open(json_path, 'r') as f:
        json_dict = json.loads(f.read())
    return len(json_dict['sensor_data'])


if __name__ == '__main__':

    hint = 'python client.py -exp 0 1 -met dl map -src data/61test'

    parser = argparse.ArgumentParser(description='client side | hint:' + hint)
    parser.add_argument('-exp', '--exp_number', type=int, nargs='+', default=[0, 1], help='number of experiments to repeat')
    parser.add_argument('-src', '--data_source', type=str, nargs=1, default=[Default.data_source], help='data source')
    parser.add_argument('-met', '--methods', type=str, nargs='+', default=Default.methods, help='methods to compare')
    parser.add_argument('-sen', '--sen_density', type=int, nargs=1, default=[Default.sen_density], help='sensor density')
    parser.add_argument('-p',   '--port', type=int, nargs=1, default=[5000], help='different port of the server holds different data')
    args = parser.parse_args()

    experimemts = args.exp_number
    data_source = args.data_source[0]
    methods = args.methods
    sensor_density = args.sen_density[0]
    port = args.port[0]

    myinput = Input(data_source=data_source, methods=methods)

    # the journal extension introduced the PU concept
    # sensor_input_dataset = mydnn_util.SensorInputDatasetTranslation(root_dir=data_source, transform=mydnn_util.tf, transform_pu=mydnn_util.tf_pu)
    
    sensor_input_dataset = mydnn_util.SensorInputDatasetTranslation(root_dir=data_source, transform=mydnn_util.tf)
    total = sensor_input_dataset.__len__()
    print('total:', total)
    myrange = range(experimemts[0], experimemts[1])
    random.seed(time.time())
    # random.seed(0)
    index = random.sample(range(total), len(myrange))
    # index = get_index_from_log('result/11.14/log')
    print('caitao', len(myrange), len(index))
    for i, idx in zip(myrange, index):
        print(i, idx)
        myinput.experiment_num = i
        myinput.image_index = idx
        myinput.num_intruder = sensor_input_dataset[idx]['target_num']
        myinput.sensor_density = get_sen_num(idx, data_source, sensor_input_dataset.sample_per_label)
        curl = "curl -d \'{}\' -H \'Content-Type: application/json\' -X POST http://{}:{}/localize"
        command = curl.format(myinput.to_json_str(), Default.server_ip, port)
        p = Popen(command, stdout=PIPE, shell=True)
        p.wait()



# python client.py -src data/205test -met map -exp 0 200 -p 5000
# python client.py -src data/205test -met map -exp 0 200 -p 5001
# python client.py -src data/205test -met splot -exp 0 5000 -p 5003
# python client.py -src data/205test -met splot -exp 0 5000 -p 5004