'''The client side, send localization request to the server
'''

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

    myinput = Input(data_source=data_source, methods=methods, sensor_density=sensor_density)

    sensor_input_dataset = mydnn_util.SensorInputDatasetTranslation(root_dir=data_source, transform=mydnn_util.tf)
    total = sensor_input_dataset.__len__()
    myrange = range(experimemts[0], experimemts[1])
    random.seed(1)
    index = random.sample(range(total), len(myrange))
    # index = get_index_from_log('result/11.14/log')
    print(len(myrange), len(index))
    for i, idx in zip(myrange, index):
        print(i, idx)
        myinput.experiment_num = i
        myinput.image_index = idx
        myinput.num_intruder = sensor_input_dataset[idx]['target_num']
        curl = "curl -d \'{}\' -H \'Content-Type: application/json\' -X POST http://{}:{}/localize"
        command = curl.format(myinput.to_json_str(), Default.server_ip, port)
        p = Popen(command, stdout=PIPE, shell=True)
        p.wait()

# python client.py -exp 0 10 -met dl map -src data/matrix-test60

    
