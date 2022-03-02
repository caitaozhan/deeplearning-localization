'''Localization server
'''

import os
import time
import argparse
import torch
import sys
import json
import numpy as np
import pickle
import copy
from sklearn import linear_model
from flask import Flask, request
from dataclasses import dataclass
from mydnn import NetTranslation5, CNN_NoTx, CNN_i, PowerPredictor5, SubtractNet3
import mydnn_util
import myplots
from input_output import Input, Output, Default, DataInfo
from utility import Utility

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
    if port == 5000 and myinput.sensor_density in [200, 400, 800, 1000]:       # ipsn: 5000 port is for varying num of intruders
        return 'hello world'
    if port == 5001 and myinput.num_intruder in [1, 2, 3, 4, 6, 7, 8, 9, 10]:  # ipsn: 5001 port is for varying sensor density
        return 'hello world'
    if port == 5003 and myinput.sensor_density in [200, 400, 800, 1000]:       # splot: 5003 port is for varying num of intruders
        return 'hello world'
    if port == 5004 and myinput.num_intruder in [1, 2, 3, 4, 6, 7, 8, 9, 10]:  # splot: 5004 port is for varying sensor density
        return 'hello world'

    if port == 5005 and myinput.sensor_density in [200, 400, 800, 1000]:       # ipsn: 5000 port is for varying num of intruders
        return 'hello world'
    if port == 5006 and myinput.sensor_density in [200, 400, 800, 1000]:       # ipsn: 5000 port is for varying num of intruders
        return 'hello world'
    if port == 5007 and myinput.sensor_density in [200, 400, 800, 1000]:       # ipsn: 5000 port is for varying num of intruders
        return 'hello world'
    if port == 5008 and myinput.sensor_density in [200, 400, 800, 1000]:       # ipsn: 5000 port is for varying num of intruders
        return 'hello world'
    if port == 5009 and myinput.sensor_density in [200, 400, 800, 1000]:       # ipsn: 5000 port is for varying num of intruders
        return 'hello world'

    if port == 4998 and (myinput.num_intruder in [1, 2, 3, 4, 6, 7, 8, 9, 10] or myinput.sensor_density != 100):  # ipsn: 4999 port is for 100 sensor density experiments
        return 'hello world'

    if port == 4999 and (myinput.num_intruder in [1, 2, 3, 4, 6, 7, 8, 9, 10] or myinput.sensor_density != 100):  # ipsn: 4999 port is for 100 sensor density experiments
        return 'hello world'

    if port == 5010:  # the ipsn data only has a fixed sensor density. It do has varying transmitters
        pass

    # Different datasets are a little different in normalization
    # The PU and ipsn are introduced in the journal extension
    # sensor_input_dataset = mydnn_util.SensorInputDatasetTranslation(root_dir=myinput.data_source, transform=mydnn_util.tf, transform_pu=mydnn_util.tf_pu)
    # sensor_input_dataset = mydnn_util.SensorInputDatasetTranslation(root_dir=myinput.data_source, transform=mydnn_util.tf)
    sensor_input_dataset = mydnn_util.SensorInputDatasetTranslation(root_dir=myinput.data_source, transform=mydnn_util.tf_ipsn)

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
        outputs.append(Output('deepmtl-simple', errors[0], falses[0], misses[0], preds[0], end-start, power_error=[]))

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
        detections = darknet(img)
        detections = non_max_suppression(detections, conf_thres=0.9, nms_thres=0.4)
        pred_xy = [server.box2xy(detections[0].numpy())]  # add a batch dimension
        preds, errors, misses, falses = mydnn_util.Metrics.localization_error_image_continuous_detection(pred_xy, y_f, indx, debug=True)
        end = time.time()
        outputs.append(Output('deepmtl-yolo', errors[0], falses[0], misses[0], preds[0], end-start, power_error=[]))

    if 'deepmtl' in myinput.methods:  # two CNN in sequence, the second CNN is object detection
        sample = sensor_input_dataset[myinput.image_index]
        X      = torch.as_tensor(sample['matrix']).unsqueeze(0).to(device)
        y_f    = np.expand_dims(sample['target_float'], 0)
        indx   = np.expand_dims(np.array(sample['index']), 0)
        start  = time.time()
        pred_matrix = translate_net(X)
        pred_matrix = pred_matrix[0][0]  # one batch, one dimension
        img = torch.stack((pred_matrix, pred_matrix, pred_matrix), axis=0) # stack 3 together
        img = resize(img, server.DETECT_IMG_SIZE).unsqueeze(0)
        detections = darknet_cust(img)
        detections = non_max_suppression(detections, conf_thres=0.8, nms_thres=0.5)
        pred_xy = [server.box2xy(detections[0].numpy())]  # add a batch dimension
        preds, errors, misses, falses = mydnn_util.Metrics.localization_error_image_continuous_detection(pred_xy, y_f, indx, debug=True)
        end = time.time()
        outputs.append(Output('deepmtl', errors[0], falses[0], misses[0], preds[0], end-start, power_error=[]))

    if 'dtxf' in myinput.methods:
        # sensor_input_dataset_regress = mydnn_util.SensorInputDatasetRegression(root_dir=myinput.data_source, grid_len=Default.grid_length, transform=mydnn_util.dtxf_tf)
        sensor_input_dataset_regress = mydnn_util.SensorInputDatasetRegression(root_dir=myinput.data_source, grid_len=Default.grid_length, transform=mydnn_util.dtxf_tf_ipsn)
        sample = sensor_input_dataset_regress[myinput.image_index]
        X      = torch.as_tensor(sample['matrix']).unsqueeze(0).to(device)
        y      = np.array(sample['target'])
        indx   = np.expand_dims(np.array(sample['index']), 0)
        start  = time.time()
        pred_ntx = cnn1(X)
        _, pred_ntx = pred_ntx.data.cpu().max(1)
        pred_ntx = pred_ntx[0]
        cnn2 = cnn2s[pred_ntx]    # select the cnn2 according to pred_ntx
        pred_loc = cnn2(X)
        pred_loc = pred_loc.data.cpu().numpy()[0]
        pred_loc = sensor_input_dataset_regress.undo_normalize(pred_loc)
        y        = sensor_input_dataset_regress.undo_normalize(y)
        pred_loc = np.reshape(pred_loc, (len(pred_loc)//2, 2))
        y        = np.reshape(y, (len(y)//2, 2))
        radius_threshold = Default.grid_length * 0.4
        error, miss, false = Utility.compute_error(pred_loc, y, radius_threshold, False)
        end = time.time()
        pred_loc = [(float(x), float(y)) for x, y in pred_loc]
        outputs.append(Output('deeptxfinder', error, false, miss, pred_loc, end-start, power_error=[]))

    if 'map' in myinput.methods:
        ll = lls[ll_index[myinput.sensor_density]]
        json_dict = server.get_json_dict(myinput.image_index, sensor_input_dataset)
        ground_truth = json_dict['ground_truth']
        sensor_data  = json_dict['sensor_data']
        sensor_outputs = np.zeros(len(sensor_data))
        for idx, rss in sensor_data.items():
            sensor_outputs[int(idx)] = rss
        true_locations, true_powers, intruders = server.parse_ground_truth(ground_truth, ll)
        if debug:
            image = sensor_input_dataset[myinput.image_index]['matrix']
            myplots.visualize_sensor_output(image, true_locations)
        start = time.time()
        pred_locations, pred_power = ll.our_localization(np.copy(sensor_outputs), intruders, myinput.experiment_num)
        end = time.time()
        pred_locations = server.pred_loc_to_center(pred_locations)
        errors, miss, false_alarm, power_errors = ll.compute_error(true_locations, true_powers, pred_locations, pred_power)
        outputs.append(Output('map', errors, false_alarm, miss, pred_locations, end-start, power_errors))

    if 'splot' in myinput.methods:
        ll = lls[ll_index[myinput.sensor_density]]
        json_dict = server.get_json_dict(myinput.image_index, sensor_input_dataset)
        ground_truth = json_dict['ground_truth']
        sensor_data  = json_dict['sensor_data']
        sensor_outputs = np.zeros(len(sensor_data))
        for idx, rss in sensor_data.items():
            sensor_outputs[int(idx)] = rss
        true_locations, true_powers, intruders = server.parse_ground_truth(ground_truth, ll)
        if debug:
            image = sensor_input_dataset[myinput.image_index]['matrix']
            myplots.visualize_sensor_output(image, true_locations)
        start = time.time()
        pred_locations = ll.splot_localization(np.copy(sensor_outputs), intruders, myinput.experiment_num)
        end = time.time()
        pred_locations = server.pred_loc_to_center(pred_locations)
        errors, miss, false_alarm = ll.compute_error2(true_locations, pred_locations)
        outputs.append(Output('splot', errors, false_alarm, miss, pred_locations, end-start, power_error=[]))

    if 'predpower' in myinput.methods:
        # step 1: deepmtl
        sample = sensor_input_dataset[myinput.image_index]
        X      = torch.as_tensor(sample['matrix']).unsqueeze(0).to(device)
        y_f    = np.expand_dims(sample['target_float'], 0)
        power_true = np.expand_dims(sample['power'], 0)
        indx   = np.expand_dims(np.array(sample['index']), 0)
        start  = time.time()
        pred_matrix = translate_net(X)
        pred_matrix = pred_matrix[0][0]  # one batch, one dimension
        img = torch.stack((pred_matrix, pred_matrix, pred_matrix), axis=0) # stack 3 together
        img = resize(img, server.DETECT_IMG_SIZE).unsqueeze(0)
        detections = darknet_cust(img)
        detections = non_max_suppression(detections, conf_thres=0.8, nms_thres=0.5)
        pred_xy = [server.box2xy(detections[0].numpy())]  # add a batch dimension
        preds, errors, misses, falses, power_dicts = mydnn_util.Metrics.localization_error_image_continuous_detection_power(pred_xy, y_f, power_true, indx, debug=False)

        # step 2: predict power & power correction
        memorize = {}
        def pred_power_helper(X, a, b):
            '''X is the full size matrix
               a and b are two integers, (a, b) is the center of the croped image, i.e. TX location
               Return the predicted power in a 21x21 grid centered at (a, b)
            '''
            if (a, b) in memorize:
                return memorize[(a, b)]
            X = X[a - 10: a + 11, b - 10: b + 11]
            X = torch.as_tensor(X).unsqueeze(0).unsqueeze(0).to(device)
            pred_power = predpower_net(X)
            pred_power = pred_power.data.cpu().numpy()[0][0]
            memorize[(a, b)] = pred_power
            return pred_power

        power_errors = []
        for pred_location, true_power in power_dicts[0].items():  # return is in batch, the batch size is one
            # step 2.1 predict power
            a, b = int(pred_location[0]), int(pred_location[1])
            X = sample['matrix'][0]
            pred_power = pred_power_helper(X, a, b)
            # print('pred_location:', pred_location, 'true_power:', true_power, 'pred_power:', pred_power, end=',  ')
            # step 2.2 power correction
            radius = 20
            closeby = []
            for pred_location2 in power_dicts[0].keys():
                c, d = int(pred_location2[0]), int(pred_location2[1])
                if (a, b) != (c, d) and np.abs(a - c) <= radius and np.abs(b - d) <= radius:
                    dist = Utility.distance(pred_location, pred_location2)
                    closeby.append((round(dist, 4), round(pred_power_helper(X, c, d), 4), round(pred_power_helper(X, c, d) / dist, 4)))
            if closeby:
                closeby_sorted = sorted(closeby)
                closeby = []
                for item in closeby_sorted:
                    closeby.append(item[0])
                    closeby.append(item[1])
                    closeby.append(item[2])
                # print('closeby', closeby, end=',  ')
                closeby.insert(0, pred_power)
                zero_padding = [0 for _ in range(25 - len(closeby))]
                closeby.extend(zero_padding)
                delta = ridgereg.predict([closeby])
                delta = round(delta[0], 4)
                corrected_power = round(pred_power - delta, 4)
                # print('delta:', delta, 'corrected_power:', corrected_power)
                power_errors.append(float(abs(corrected_power - true_power)))
            else:
                power_errors.append(float(abs(pred_power - true_power)))
                # print()

        end = time.time()
        outputs.append(Output('predpower', errors[0], falses[0], misses[0], preds[0], end-start, power_errors))

    if 'predpower_nocorrect' in myinput.methods:
        # step 1: deepmtl
        sample = sensor_input_dataset[myinput.image_index]
        X      = torch.as_tensor(sample['matrix']).unsqueeze(0).to(device)
        y_f    = np.expand_dims(sample['target_float'], 0)
        power_true = np.expand_dims(sample['power'], 0)
        indx   = np.expand_dims(np.array(sample['index']), 0)
        start  = time.time()
        pred_matrix = translate_net(X)
        pred_matrix = pred_matrix[0][0]  # one batch, one dimension
        img = torch.stack((pred_matrix, pred_matrix, pred_matrix), axis=0) # stack 3 together
        img = resize(img, server.DETECT_IMG_SIZE).unsqueeze(0)
        detections = darknet_cust(img)
        detections = non_max_suppression(detections, conf_thres=0.8, nms_thres=0.5)
        pred_xy = [server.box2xy(detections[0].numpy())]  # add a batch dimension
        preds, errors, misses, falses, power_dicts = mydnn_util.Metrics.localization_error_image_continuous_detection_power(pred_xy, y_f, power_true, indx, debug=False)

        # step 2: predict power & power correction
        memorize = {}
        def pred_power_helper(X, a, b):
            '''X is the full size matrix
               a and b are two integers, (a, b) is the center of the croped image, i.e. TX location
               Return the predicted power in a 21x21 grid centered at (a, b)
            '''
            if (a, b) in memorize:
                return memorize[(a, b)]
            X = X[a - 10: a + 11, b - 10: b + 11]
            X = torch.as_tensor(X).unsqueeze(0).unsqueeze(0).to(device)
            pred_power = predpower_net(X)
            pred_power = pred_power.data.cpu().numpy()[0][0]
            memorize[(a, b)] = pred_power
            return pred_power

        power_errors = []
        for pred_location, true_power in power_dicts[0].items():  # return is in batch, the batch size is one
            # step 2.1 predict power
            a, b = int(pred_location[0]), int(pred_location[1])
            X = sample['matrix'][0]
            pred_power = pred_power_helper(X, a, b)
            # print('pred_location:', pred_location, 'true_power:', true_power, 'pred_power:', pred_power, end=',  ')
            power_errors.append(float(abs(pred_power - true_power)))
            # print()

        end = time.time()
        outputs.append(Output('predpower_nocorrect', errors[0], falses[0], misses[0], preds[0], end-start, power_errors))

    if 'deepmtl_auth' in myinput.methods:  # localize all and then subtract
        sample = sensor_input_dataset[myinput.image_index]
        X      = torch.as_tensor(sample['matrix_auth']).unsqueeze(0).to(device)   # the matrix (sensor reading image) with invaders + authorized users
        y_f    = np.expand_dims(sample['target_float'], 0)                        # ground truth location for the invaders
        y_auth = sample['target_auth_float']                   # ground truth location for the authorized users
        indx   = np.expand_dims(np.array(sample['index']), 0)
        start  = time.time()
        pred_matrix = translate_net(X)
        pred_matrix = pred_matrix[0][0]  # one batch, one dimension
        img = torch.stack((pred_matrix, pred_matrix, pred_matrix), axis=0) # stack 3 together
        img = resize(img, server.DETECT_IMG_SIZE).unsqueeze(0)
        detections = darknet_cust(img)
        detections = non_max_suppression(detections, conf_thres=0.8, nms_thres=0.4)
        pred_xy = server.box2xy(detections[0].numpy())
        server.remove_authorized(pred_xy, y_auth)
        pred_xy = [pred_xy]   # add a batch dimension
        preds, errors, misses, falses = mydnn_util.Metrics.localization_error_image_continuous_detection(pred_xy, y_f, indx, debug=False)
        end = time.time()
        outputs.append(Output('deepmtl_auth', errors[0], falses[0], misses[0], preds[0], end-start, []))

    if 'deepmtl_auth_subtractpower' in myinput.methods:   # subtract the PU power via CNN and then localize
        sample = sensor_input_dataset[myinput.image_index]
        X      = torch.as_tensor(sample['two_sheet']).unsqueeze(0).to(device)
        y_f    = np.expand_dims(sample['target_float'], 0)
        indx   = np.expand_dims(np.array(sample['index']), 0)
        start = time.time()
        subtracted = subtract_net(X)
        pred_matrix = translate_net(subtracted)
        pred_matrix = pred_matrix[0][0]
        img = torch.stack((pred_matrix, pred_matrix, pred_matrix), axis=0)
        img = resize(img, server.DETECT_IMG_SIZE).unsqueeze(0)
        detections = darknet_cust(img)
        detections = non_max_suppression(detections, conf_thres=0.85, nms_thres=0.4)
        pred_xy = [server.box2xy(detections[0].numpy())]  # add a batch dimension
        preds, errors, misses, falses = mydnn_util.Metrics.localization_error_image_continuous_detection(pred_xy, y_f, indx, debug=False)
        end = time.time()
        outputs.append(Output('deepmtl_auth_subtractpower', errors[0], falses[0], misses[0], preds[0], end-start, []))

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
                    # true_powers.append(0)  # all TX are well calibrated
                    true_powers.append(value)  # all TX are well calibrated
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

    def remove_authorized(self, pred_locations, authorized):
        '''remove the authorized user from the predicted locations
        Args:
            pred_locations -- np.ndarray, n=3, [tx_index, x, y]    -- update in place
            authorized     -- np.ndarray, n=3, [auth_index, x, y]
        '''
        if len(pred_locations) == 0:
            return
        # print('\nAuthorized are:', authorized)
        distances = np.zeros((len(authorized), len(pred_locations)))
        for i in range(len(authorized)):
            for j in range(len(pred_locations)):
                distances[i, j] = np.sqrt((authorized[i][0] - pred_locations[j][0]) ** 2 + (authorized[i][1] - pred_locations[j][1]) ** 2)

        k = 0
        matches = []
        pred_loc_remove = []
        while k < min(len(authorized), len(pred_locations)):
            min_distance_index = np.argmin(distances)
            i = min_distance_index // len(pred_locations)
            j = min_distance_index % len(pred_locations)
            matches.append((i, j))          # authoried i matches predicted j
            pred_loc_remove.append(pred_locations[j])
            distances[i, :] = np.inf
            distances[:, j] = np.inf
            k += 1

        for loc in pred_loc_remove:
            pred_locations.remove(loc)


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
    parser.add_argument('-src', '--data_source', type=str, nargs=1, default=[None], help='the testing data source')
    parser.add_argument('-p', '--port', type=int, nargs=1, default=[5000], help='the port number')
    parser.add_argument('-pl', '--plus', action='store_true', default=False, help='if given, do the no retrain for second step experiment')
    args = parser.parse_args()

    data_source = args.data_source[0]
    port = args.port[0]
    if args.plus:
        data_source += '_plus'

    data = DataInfo.naive_factory(data_source=data_source)
    # 1: init server utilities
    date = '3.1'                                                 # 1
    output_dir = f'result/{date}'
    # output_file = f'splat-dtxf-{port}'                                        # 2
    # output_file = f'splat-map-{port}'                                        # 2
    # output_file = f'splat-splot-{port}'                                        # 2
    # output_file = f'logdistance-all-100_sendensity-{port}'                                        # 2
    # output_file = f'splat-all-100_sendensity-{port}'                                        # 2
    output_file = f'ipsn-all-{port}'                                        # 2
    # output_file = f'splat-deepmtl-{port}'                                        # 2
    # output_file = f'splat-deepmtl_auth_subtractpower3-{port}-conf=0.85,nms=0.4'                                        # 2
    # output_file = f'logdistance-deepmtl.predpower-{port}'
    # output_file = f'logdistance-deepmtl.predpower_and_nocorrect-{port}'
    # if args.plus:
    #     output_file += '_plus'                                  # for the journal: no replace part 2
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

    # *** FOR Power Estimation only, init both predpower and ridgereg ***
    # predpower_net = PowerPredictor5()
    # predpower_net.load_state_dict(torch.load(data.predpower_net))
    # predpower_net = predpower_net.to(device)
    # predpower_net.eval()
    # ridgereg = pickle.load(open(data.power_corrector, 'rb'))

    # *** FOR subtracting authorized user power only ***
    # subtract_net = SubtractNet3()
    # subtract_net.load_state_dict(torch.load(data.subtract_net))
    # subtract_net = subtract_net.to(device)
    # subtract_net.eval()


    # 3.1: init the darknet
    # darknet = Darknet(data.yolo_def, img_size=server.DETECT_IMG_SIZE).to(device)
    # darknet.load_state_dict(torch.load(data.yolo_weights))
    # darknet.eval()

    # 4: init MAP* (and SPLOT)
    # grid_len = 100
    # debug = False                                                   # 3
    # # case = 'lognormal3'                                             # 4
    # case = 'splat3'                                               # 4
    # lls = []
    # ll_index = {100:0, 200:1, 400:2, 600:3, 800:4, 1000:5}
    # for i in range(len(data.ipsn_cov_list)):
    #     ll = Localization(grid_len=grid_len, case=case, debug=debug)
    #     ll.init_data(data.ipsn_cov_list[i], data.ipsn_sensors_list[i], data.ipsn_hypothesis_list[i], None)
    #     lls.append(ll)

    # ll = Localization(grid_len=grid_len, case=case, debug=debug)
    # ll.init_data(data.ipsn_cov_list[2], data.ipsn_sensors_list[2], data.ipsn_hypothesis_list[2], None)
    # for i in range(len(data.ipsn_cov_list)):
    #     lls.append(ll)

    # 5 init deeptxfinder
    device = torch.device('cuda')
    max_ntx = 10
    cnn1  = CNN_NoTx(max_ntx)
    cnn1.load_state_dict(torch.load(data.dtxf_cnn1))
    cnn1 = cnn1.to(device)
    cnn1.eval()
    cnn2s = []
    cnn2_template = data.dtxf_cnn2_template
    for i in range(max_ntx):
        num_ntx = i + 1
        cnn2 = CNN_i(num_ntx)
        cnn2.load_state_dict(torch.load(cnn2_template.format(num_ntx)))
        cnn2 = cnn2.to(device)
        cnn2.eval()
        cnn2s.append(cnn2)

    # 6: start the web server
    print('process time:', time.process_time())
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)



# python server.py -src data/205test -p 5000
# python server.py -src data/205test -p 5001
# python server.py -src data/205test -p 5003
# python server.py -src data/205test -p 5004