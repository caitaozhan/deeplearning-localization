'''Localization server
'''

import os
import time
import argparse
import torch
import sys
import json
import numpy as np
from flask import Flask, request
from dataclasses import dataclass
from mydnn import NetTranslation5
import mydnn_util
import myplots
from input_output import Input, Output, Default, DataInfo

# The IPSN20 algo
try:
    sys.path.append('../Localization')
    from localize import Localization
    from plots import visualize_localization
except Exception as e:
    raise(e)

# The dl3 algo (image translation + detection)
try:
    sys.path.append('../PyTorch-YOLOv3')
    from models import Darknet
    from utils.utils import non_max_suppression
    from utils.datasets import resize
except Exception as e:
    raise(e)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world!'

@app.route('/localize', methods=['POST'])
def localize():
    '''localize
    '''
    myinput = Input.from_json_dict(request.get_json())
    sensor_input_dataset = mydnn_util.SensorInputDatasetTranslation(root_dir=myinput.data_source, transform=mydnn_util.tf)
    outputs = []
    if 'deepmtl-simple' in myinput.methods:  # two CNN in sequence, the second CNN is object detection
        sample = sensor_input_dataset[myinput.image_index]
        X      = torch.as_tensor(sample['matrix']).unsqueeze(0).to(device)
        y_f    = np.expand_dims(sample['target_float'], 0)
        indx   = np.expand_dims(np.array(sample['index']), 0)
        start = time.time()
        pred_matrix = translate_net(X)
        pred_matrix = pred_matrix.data.cpu().numpy()
        preds, errors, misses, falses = mydnn_util.Metrics.localization_error_image_continuous_simple(pred_matrix, y_f, indx, Default.grid_length, peak_threshold=2, size=3, debug=True)
        end = time.time()
        outputs.append(Output('deepmtl-simple', errors[0], falses[0], misses[0], preds[0], end-start))

    if 'deepmtl-yolo' in myinput.methods:  # two CNN in sequence, the second CNN is object detection
        sample = sensor_input_dataset[myinput.image_index]
        X      = torch.as_tensor(sample['matrix']).unsqueeze(0).to(device)
        y_f    = np.expand_dims(sample['target_float'], 0)
        indx   = np.expand_dims(np.array(sample['index']), 0)
        start = time.time()
        pred_matrix = translate_net(X)
        pred_matrix = pred_matrix[0][0]  # one batch, one dimension
        img = torch.stack((pred_matrix, pred_matrix, pred_matrix), axis=0) # stack 3 together
        img = resize(img, server.DETECT_IMG_SIZE).unsqueeze(0)
        # detections = darknet(img)
        detections = non_max_suppression(detections, conf_thres=0.9, nms_thres=0.4)
        pred_xy = [server.box2xy(detections[0].numpy())]  # add a batch dimension
        preds, errors, misses, falses = mydnn_util.Metrics.localization_error_image_continuous_detection(pred_xy, y_f, indx, debug=True)
        end = time.time()
        outputs.append(Output('deepmtl-yolo', errors[0], falses[0], misses[0], preds[0], end-start))

    if 'deepmtl' in myinput.methods:  # two CNN in sequence, the second CNN is object detection
        sample = sensor_input_dataset[myinput.image_index]
        X      = torch.as_tensor(sample['matrix']).unsqueeze(0).to(device)
        y_f    = np.expand_dims(sample['target_float'], 0)
        indx   = np.expand_dims(np.array(sample['index']), 0)
        start = time.time()
        pred_matrix = translate_net(X)
        pred_matrix = pred_matrix[0][0]  # one batch, one dimension
        img = torch.stack((pred_matrix, pred_matrix, pred_matrix), axis=0) # stack 3 together
        img = resize(img, server.DETECT_IMG_SIZE).unsqueeze(0)
        detections = darknet_cust(img)
        detections = non_max_suppression(detections, conf_thres=0.9, nms_thres=0.5)
        pred_xy = [server.box2xy(detections[0].numpy())]  # add a batch dimension
        preds, errors, misses, falses = mydnn_util.Metrics.localization_error_image_continuous_detection(pred_xy, y_f, indx, debug=True)
        end = time.time()
        outputs.append(Output('deepmtl', errors[0], falses[0], misses[0], preds[0], end-start))

    if 'map' in myinput.methods:
        json_dict = server.get_json_dict(myinput.image_index, sensor_input_dataset)
        ground_truth = json_dict['ground_truth']
        sensor_data = json_dict['sensor_data']
        sensor_outputs = np.zeros(len(sensor_data))
        for idx, rss in sensor_data.items():
            sensor_outputs[int(idx)] = rss
        true_locations, true_powers, intruders = server.parse_ground_truth(ground_truth, ll)
        image = sensor_input_dataset[myinput.image_index]['matrix']
        myplots.visualize_sensor_output(image, true_locations)
        start = time.time()
        pred_locations, pred_power = ll.our_localization(np.copy(sensor_outputs), intruders, myinput.experiment_num)
        end = time.time()
        pred_locations = server.pred_loc_to_center(pred_locations)
        errors, miss, false_alarm, power_errors = ll.compute_error(true_locations, true_powers, pred_locations, pred_power)
        outputs.append(Output('map', errors, false_alarm, miss, pred_locations, end-start))

    if 'splot' in myinput.methods:
        pass

    if 'dtxf' in myinput.methods:
        pass

    server.log(myinput, outputs)
    return 'hello world'


class Server:
    '''Misc things to support the server running
    '''
    DETECT_IMG_SIZE = 416

    def __init__(self, outout_dir, output_file):
        self.output = self.init_output(outout_dir, output_file)

    def init_output(self, output_dir, output_file):
        '''set up output file
        Args:
            output_dir  -- str
            output_file -- str
        Return:
            io.TextIOWrapper
        '''
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)
        return open(output_dir + '/' + output_file, 'a')

    def log(self, myinput, outputs):
        '''log the results
        Args:
            myinput -- Input
            outputs -- List[Output]
        '''
        self.output.write(f'{myinput.log()}\n')
        for output in outputs:
            self.output.write(f'{output.log()}\n')
        self.output.write('\n')
        self.output.flush()

    def get_json_dict(self, image_index, sensor_input_dataset):
        '''get the json data
        Args:
            image_index -- int
            sensor_input_dataset -- SensorInputDatasetTranslation
        Return:
            dict
        '''
        folder = int(image_index/sensor_input_dataset.sample_per_label)
        folder = format(folder, '06d')
        json_name = str(image_index%sensor_input_dataset.sample_per_label) + '.json'
        json_path = os.path.join(sensor_input_dataset.root_dir, folder, json_name)
        with open(json_path, 'r') as f:
            jsondict = json.loads(f.read())
        return jsondict

    def parse_ground_truth(self, ground_truth, ll):
        '''parse the ground truth from the client
        Args:
            ground_truth -- {...} eg. {'T1': {'location': [9.5, 5.5], 'gain': '50'}}
            ll           -- Localization
        Return:
            true_locations -- list<(float, float)>
            true_powers    -- list<float>
            intruders      -- list<Transmitter>
        '''
        grid_len = ll.grid_len
        true_locations, true_powers = [], []
        intruders = []
        for tx, truth in sorted(ground_truth.items()):
            for key, value in truth.items():
                if key == 'location':
                    one_d_index = (int(value[0])*grid_len + int(value[1]))
                    two_d_index = (value[0], value[1])
                    true_locations.append(two_d_index)
                    intruders.append(ll.transmitters[one_d_index])
                elif key == 'gain':
                    # train_power = self.tx_calibrate[tx]
                    # true_powers.append(float(value) - train_power)
                    true_powers.append(0)  # all TX are well calibrated
                else:
                    raise Exception('key = {} invalid!'.format(key))
        return true_locations, true_powers, intruders

    def pred_loc_to_center(self, pred_locations):
        '''Make the predicted locations be the center of the predicted grid
        Args:
            pred_locations -- list<tuple<int, int>>
        Return:
            list<tuple<int, int>>
        '''
        pred_center = []
        for pred in pred_locations:
            center = (pred[0] + 0.5, pred[1] + 0.5)
            pred_center.append(center)
        return pred_center

    def box2xy(self, detections):
        '''object detections model returns bounding boxes. transform this into (x, y) by computing the center
        '''
        pred_locs = []
        for detect in detections:
            x1 = detect[0]
            y1 = detect[1]
            x2 = detect[2]
            y2 = detect[3]
            x = (x1 + x2) / 2 * (Default.grid_length / server.DETECT_IMG_SIZE)  # assume no padding (every image is square)
            y = (y1 + y2) / 2 * (Default.grid_length / server.DETECT_IMG_SIZE)
            pred_locs.append((y, x))                                            # note that in YOLO v3, the (x, y) are opposite to my (x, y)
        return pred_locs

# if __name__ == 'server':

    # hint = 'python server.py -src data/60test'
    # parser = argparse.ArgumentParser(description='Server side. ' + hint)
    # parser.add_argument('-src', '--data_source', type=str,  nargs=1, default=[None], help='the testing data source')
    # args = parser.parse_args()

    # data_source = args.data_source[0]

    # data = DataInfo.naive_factory(data_source=data_source)
    # # 1: init server utilities
    # date = '11.13'
    # output_dir = f'result/{date}'
    # output_file = 'log'
    # server = Server(output_dir, output_file)

    # # 2: init deep learning model
    # # max_ntx = 5
    # # path1 = data.dl_model1
    # # path2 = data.dl_model2
    # # device = torch.device('cuda')
    # # model1 = NetTranslation()
    # # model1.load_state_dict(torch.load(path1))
    # # model1 = model1.to(device)
    # # model2 = NetNumTx(max_ntx)
    # # model2.load_state_dict(torch.load(path2))
    # # model2.to(device)
    # # model2 = model2.to(device)
    # # model1.eval()
    # # model2.eval()
    # # print('process time', time.process_time())

    # # 3: init IPSN20
    # ll = Localization(data.ipsn_cov, data.ipsn_sensors, data.ipsn_hypothesis, None)
    # print('caitao')


if __name__ == '__main__':

    hint = 'python server.py -src data/205test'
    parser = argparse.ArgumentParser(description='Server side. ' + hint)
    parser.add_argument('-src', '--data_source', type=str,  nargs=1, default=[None], help='the testing data source')
    args = parser.parse_args()

    data_source = args.data_source[0]

    data = DataInfo.naive_factory(data_source=data_source)
    # 1: init server utilities
    date = '12.9'                                                 # 1
    output_dir = f'result/{date}'
    output_file = 'log-deepmtl-simple2'                            # 2
    server = Server(output_dir, output_file)

    # 2: init image to image translation model
    device = torch.device('cuda')

    translate_net = NetTranslation5()
    translate_net.load_state_dict(torch.load(data.translate_net))
    translate_net = translate_net.to(device)
    translate_net.eval()

    # 3: init the darknet_cust
    darknet_cust = Darknet(data.yolocust_def, img_size=server.DETECT_IMG_SIZE).to(device)
    darknet_cust.load_state_dict(torch.load(data.yolocust_weights))
    darknet_cust.eval()

    # 4: init IPSN20
    # grid_len = 100
    # debug = False                                                   # 3
    # case = 'lognormal2'
    # lls = []
    # for i in range(len(data.ipsn_cov_list)):
    #     ll = Localization(grid_len=grid_len, case=case, debug=debug)
    #     ll.init_data(data.ipsn_cov_list[i], data.ipsn_sensors_list[i], data.ipsn_hypothesis_list[i], None)
    #     lls.append(ll)

    # 6 init SPLOT TODO

    # 7 init deeptxfinder TODO


    # 5: start the web server
    print('process time:', time.process_time())
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

