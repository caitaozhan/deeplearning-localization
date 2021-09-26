'''
Encapsulate the input, output variables, as well as some default configurations
'''
from dataclasses import dataclass
from typing import List, Dict
import json
import numpy as np


@dataclass
class Default:
    '''some default configurations
    '''
    alpha            = 3.5     # the slope of wireless signal depreciation
    std              = 1       # the standard deviation of the zero mean shadowing
    data_source      = 'log-distance'
    methods          = ['dl']
    sen_density      = 600
    num_intruder     = 5
    grid_length      = 100
    cell_length      = 10
    random_seed      = 0
    noise_floor      = -80
    power            = 10
    cell_percentage  = 1
    sample_per_label = 10
    root_dir         = 'data/images-1'  # the root directory of the data
    num_tx           = 1
    error_threshold  = 0.2
    min_dist         = 1
    max_dist         = None
    edge             = 2
    methods          = ['deepmtl', 'deepmtl-simple', 'deepmtl-yolo', 'map', 'splot', 'dtxf']
    server_ip        = '0.0.0.0'


@dataclass
class Input:
    '''encapsulate the input variables
    '''
    methods: List[str]
    experiment_num: int = -1
    num_intruder: int   = Default.num_intruder
    data_source: str    = Default.data_source
    sensor_density: int = Default.sen_density
    image_index: int     = -1   # client send an index, and the server read the data locally according the index (client and server are on the same machine)

    def to_json_str(self):
        '''return json formated string
        Return:
            str
        '''
        inputdict = {
            'experiment_num':self.experiment_num,
            'num_intruder':self.num_intruder,
            'image_index':self.image_index,
            'methods':self.methods,
            'data_source':self.data_source,
            'sensor_density':self.sensor_density
        }
        return json.dumps(inputdict)

    @classmethod
    def from_json_str(cls, json_str):
        '''Init an Input object from json string
        Args:
            json_str -- str
        Return:
            Input
        '''
        inputdict = json.loads(json_str)
        return cls.from_json_dict(inputdict)

    @classmethod
    def from_json_dict(cls, json_dict):
        '''Init an Input object from json dictionary
        Args:
            json_dict -- dict
        Return:
            Input
        '''
        myinput = cls([])
        myinput.experiment_num = json_dict['experiment_num']
        myinput.num_intruder   = json_dict['num_intruder']
        myinput.image_index    = json_dict['image_index']
        myinput.methods        = json_dict['methods']
        myinput.data_source    = json_dict['data_source']
        myinput.sensor_density = json_dict['sensor_density']
        return myinput

    def log(self):
        '''log'''
        return self.to_json_str()


@dataclass
class IpsnInput:
    '''input data of IPSN20 localization method
    '''
    ground_truth: Dict
    sensor_data:  Dict

    def to_json_str(self):
        '''return json formated string
        Return:
            str
        '''
        inputdict = {
            "ground_truth": self.ground_truth,
            "sensor_data":  self.sensor_data,
        }
        return json.dumps(inputdict)

    @classmethod
    def from_file(cls, file):
        '''init from a file
        '''
        with open(file, 'r') as f:
            line = f.readline()
            json_dict = json.loads(line)
            return cls(json_dict['ground_truth'], json_dict['sensor_data'])


@dataclass
class Output:
    '''encapsulate the output variables
    '''
    method: List[str]
    error: List[float]            # error of the detected TX
    false_alarm: int
    miss: int
    preds: List
    time: float
    power_error: List[float]

    def get_metric(self, metric):
        '''get the evaluation metrics'''
        if metric == 'error':
            return round(np.mean(self.error), 3)
        elif metric == 'miss':
            return self.miss
        elif metric == 'false_alarm':
            return self.false_alarm
        elif metric == 'time':
            return self.time
        elif metric == 'power_error':
            return round(np.mean(np.abs(self.power_error)), 3)
        else:
            raise Exception('unknown metrics')

    def get_pred_len(self):
        '''Get the number of predicted TX
        '''
        size = len(self.preds)
        return size if size != 0 else 1  # if len(self.preds) equals 0, then the false alarm rate will be zero anyway because the false alarm (numerator is 0)

    def to_json_str(self):
        '''return json formated string
        Return:
            str
        '''
        self.preds = [(round(x, 2), round(y, 2)) for x, y in self.preds]
        self.power_error = [round(x, 4) for x in self.power_error]
        self.time = round(self.time, 5)
        outputdict = {
            "method":self.method,
            "error":self.error,
            "false_alarm":self.false_alarm,
            "miss":self.miss,
            "preds":self.preds,
            "time":self.time,
            "power_error":self.power_error
        }
        return json.dumps(outputdict)

    @classmethod
    def from_json_str(cls, json_str):
        '''Init an Output object from json
        Args:
            json_str -- str
        Return:
            Output
        '''
        outputdict = json.loads(json_str)
        return cls.from_json_dict(outputdict)

    @classmethod
    def from_json_dict(cls, json_dict):
        '''Init an Output object from json dictionary
        Args:
            json_dict -- dict
        Return:
            Output
        '''
        method = json_dict['method']
        error = json_dict['error']
        false_alarm = json_dict['false_alarm']
        miss = json_dict['miss']
        preds = json_dict['preds']
        time = json_dict['time']
        power_error = json_dict['power_error']
        return cls(method, error, false_alarm, miss, preds, time, power_error)

    def log(self):
        return self.to_json_str()


@dataclass
class DataInfo:
    '''the data set used for training and testing
    '''
    max_ntx: int
    test_data: str
    train_data: str
    ipsn_cov_list: List            # there are five ipsn dataset for five differnent set of sensors
    ipsn_sensors_list: List
    ipsn_hypothesis_list: List
    translate_net: str        # image translation
    yolocust_def: str         # our yolo cust model definition
    yolocust_weights: str     # our yolo cust model weights
    yolo_def: str             # yolo model definition  
    yolo_weights: str         # yolo model weights
    dtxf_cnn1: str
    dtxf_cnn2_template: str

    @classmethod
    def naive_factory(cls, data_source):
        '''factory'''
        if data_source == 'data/205test':  # the log-distancec based model
            max_ntx = 10
            test_data  = 'data/205test'
            train_data = 'data/205train'
            ipsn_cov_list  = ['data/200test-ipsn/cov',        'data/201test-ipsn/cov',        'data/202test-ipsn/cov',        'data/203test-ipsn/cov',        'data/204test-ipsn/cov']
            ipsn_sen_list  = ['data/200test-ipsn/sensors',    'data/201test-ipsn/sensors',    'data/202test-ipsn/sensors',    'data/203test-ipsn/sensors',    'data/204test-ipsn/sensors']
            ipsn_hypo_list = ['data/200test-ipsn/hypothesis', 'data/201test-ipsn/hypothesis', 'data/202test-ipsn/hypothesis', 'data/203test-ipsn/hypothesis', 'data/204test-ipsn/hypothesis']
            translate_net  = 'model/model1-12.8-net5-norm-32.pt'
            yolocust_def     = '../PyTorch-YOLOv3/config/yolov3-custom.cfg'
            yolocust_weights = '../PyTorch-YOLOv3/checkpoints_logdistance/yolov3_ckpt_5.pth'
            yolo_def         = '../PyTorch-YOLOv3/config/yolov3-custom-class.cfg'
            yolo_weights     = '../PyTorch-YOLOv3/checkpoints_logdistance_class/yolov3_ckpt_5.pth'
            dtxf_cnn1        =   'model_dtxf/12.12-cnn1-logdist.pt'
            dtxf_cnn2_template = 'model_dtxf/12.12-cnn2-logdist_{}.pt'
            return cls(max_ntx, test_data, train_data, ipsn_cov_list, ipsn_sen_list, ipsn_hypo_list, \
                       translate_net, yolocust_def, yolocust_weights, yolo_def, yolo_weights, dtxf_cnn1, dtxf_cnn2_template)

        if data_source == 'data/305test':  # the splat model
            max_ntx = 10
            test_data  = 'data/305test'
            train_data = 'data/305train'
            ipsn_cov_list  = ['data/300test-ipsn/cov',        'data/301test-ipsn/cov',        'data/302test-ipsn/cov',        'data/303test-ipsn/cov',        'data/304test-ipsn/cov']
            ipsn_sen_list  = ['data/300test-ipsn/sensors',    'data/301test-ipsn/sensors',    'data/302test-ipsn/sensors',    'data/303test-ipsn/sensors',    'data/304test-ipsn/sensors']
            ipsn_hypo_list = ['data/300test-ipsn/hypothesis', 'data/301test-ipsn/hypothesis', 'data/302test-ipsn/hypothesis', 'data/303test-ipsn/hypothesis', 'data/304test-ipsn/hypothesis']
            translate_net  = 'model/model1-12.13-net5-norm-32-splat.pt'
            yolocust_def     = '../PyTorch-YOLOv3/config/yolov3-custom.cfg'
            yolocust_weights = '../PyTorch-YOLOv3/checkpoints/yolov3_ckpt_5.pth'
            yolo_def         = '../PyTorch-YOLOv3/config/yolov3-custom-class.cfg'
            yolo_weights     = '../PyTorch-YOLOv3/checkpoints_logdistance_class/yolov3_ckpt_5.pth'
            dtxf_cnn1        =   'model_dtxf/12.13-cnn1-splat.pt'
            dtxf_cnn2_template = 'model_dtxf/12.13-cnn2-splat_{}.pt'
            return cls(max_ntx, test_data, train_data, ipsn_cov_list, ipsn_sen_list, ipsn_hypo_list, \
                       translate_net, yolocust_def, yolocust_weights, yolo_def, yolo_weights, dtxf_cnn1, dtxf_cnn2_template)

        # don't need to retrain the second part (YOLO): first part log-distance, second part splat
        if data_source == 'data/205test_plus':  # the log-distancec based model
            max_ntx = 10
            translate_net    = 'model/model1-12.8-net5-norm-32.pt'
            yolocust_def     = '../PyTorch-YOLOv3/config/yolov3-custom.cfg'
            yolocust_weights = '../PyTorch-YOLOv3/checkpoints/yolov3_ckpt_5.pth'
            # below are useless
            test_data  = 'data/205test'
            train_data = 'data/205train'
            ipsn_cov_list  = ['data/200test-ipsn/cov',        'data/201test-ipsn/cov',        'data/202test-ipsn/cov',        'data/203test-ipsn/cov',        'data/204test-ipsn/cov']
            ipsn_sen_list  = ['data/200test-ipsn/sensors',    'data/201test-ipsn/sensors',    'data/202test-ipsn/sensors',    'data/203test-ipsn/sensors',    'data/204test-ipsn/sensors']
            ipsn_hypo_list = ['data/200test-ipsn/hypothesis', 'data/201test-ipsn/hypothesis', 'data/202test-ipsn/hypothesis', 'data/203test-ipsn/hypothesis', 'data/204test-ipsn/hypothesis']
            yolo_def         = '../PyTorch-YOLOv3/config/yolov3-custom-class.cfg'
            yolo_weights     = '../PyTorch-YOLOv3/checkpoints_logdistance_class/yolov3_ckpt_5.pth'
            dtxf_cnn1        =   'model_dtxf/12.12-cnn1-logdist.pt'
            dtxf_cnn2_template = 'model_dtxf/12.12-cnn2-logdist_{}.pt'
            return cls(max_ntx, test_data, train_data, ipsn_cov_list, ipsn_sen_list, ipsn_hypo_list, \
                       translate_net, yolocust_def, yolocust_weights, yolo_def, yolo_weights, dtxf_cnn1, dtxf_cnn2_template)

        # don't need to retrain the second part (YOLO): first part splat, second part log-distance
        if data_source == 'data/305test_plus':  # the splat model
            translate_net    = 'model/model1-12.13-net5-norm-32-splat.pt'
            yolocust_def     = '../PyTorch-YOLOv3/config/yolov3-custom.cfg'
            yolocust_weights = '../PyTorch-YOLOv3/checkpoints_logdistance/yolov3_ckpt_5.pth'
            # below are useless
            max_ntx = 10
            test_data  = 'data/305test'
            train_data = 'data/305train'
            ipsn_cov_list  = ['data/300test-ipsn/cov',        'data/301test-ipsn/cov',        'data/302test-ipsn/cov',        'data/303test-ipsn/cov',        'data/304test-ipsn/cov']
            ipsn_sen_list  = ['data/300test-ipsn/sensors',    'data/301test-ipsn/sensors',    'data/302test-ipsn/sensors',    'data/303test-ipsn/sensors',    'data/304test-ipsn/sensors']
            ipsn_hypo_list = ['data/300test-ipsn/hypothesis', 'data/301test-ipsn/hypothesis', 'data/302test-ipsn/hypothesis', 'data/303test-ipsn/hypothesis', 'data/304test-ipsn/hypothesis']
            yolo_def         = '../PyTorch-YOLOv3/config/yolov3-custom-class.cfg'
            yolo_weights     = '../PyTorch-YOLOv3/checkpoints_logdistance_class/yolov3_ckpt_5.pth'
            dtxf_cnn1        =   'model_dtxf/12.13-cnn1-splat.pt'
            dtxf_cnn2_template = 'model_dtxf/12.13-cnn2-splat_{}.pt'
            return cls(max_ntx, test_data, train_data, ipsn_cov_list, ipsn_sen_list, ipsn_hypo_list, \
                       translate_net, yolocust_def, yolocust_weights, yolo_def, yolo_weights, dtxf_cnn1, dtxf_cnn2_template)
        
        if data_source == 'data/605test':  # the logdistance model
            test_data  = 'data/605test'
            train_data = 'data/605train'
            ipsn_cov_list  = ['data/600test-ipsn/cov',        'data/601test-ipsn/cov',        'data/602test-ipsn/cov',        'data/603test-ipsn/cov',        'data/604test-ipsn/cov']
            ipsn_sen_list  = ['data/600test-ipsn/sensors',    'data/601test-ipsn/sensors',    'data/602test-ipsn/sensors',    'data/603test-ipsn/sensors',    'data/604test-ipsn/sensors']
            ipsn_hypo_list = ['data/600test-ipsn/hypothesis', 'data/601test-ipsn/hypothesis', 'data/602test-ipsn/hypothesis', 'data/603test-ipsn/hypothesis', 'data/604test-ipsn/hypothesis']
            yolocust_def     = '../PyTorch-YOLOv3/config/yolov3-custom.cfg'
            yolocust_weights = '../PyTorch-YOLOv3/checkpoints_logdistance/yolov3_ckpt_5.pth'  # to be updated
            translate_net    = 'model/model1-12.13-net5-norm-32-splat.pt'                     # to be updated
            # below are useless
            max_ntx          = 10
            yolo_def         = '../PyTorch-YOLOv3/config/yolov3-custom-class.cfg'
            yolo_weights     = '../PyTorch-YOLOv3/checkpoints_logdistance_class/yolov3_ckpt_5.pth'
            dtxf_cnn1        =   'model_dtxf/12.13-cnn1-splat.pt'
            dtxf_cnn2_template = 'model_dtxf/12.13-cnn2-splat_{}.pt'
            return cls(max_ntx, test_data, train_data, ipsn_cov_list, ipsn_sen_list, ipsn_hypo_list, \
                       translate_net, yolocust_def, yolocust_weights, yolo_def, yolo_weights, dtxf_cnn1, dtxf_cnn2_template)

        if data_source == 'data/705test':  # the splat model
            test_data  = 'data/705test'
            train_data = 'data/705train'
            ipsn_cov_list  = ['data/700test-ipsn/cov',        'data/701test-ipsn/cov',        'data/702test-ipsn/cov',        'data/703test-ipsn/cov',        'data/704test-ipsn/cov']
            ipsn_sen_list  = ['data/700test-ipsn/sensors',    'data/701test-ipsn/sensors',    'data/702test-ipsn/sensors',    'data/703test-ipsn/sensors',    'data/704test-ipsn/sensors']
            ipsn_hypo_list = ['data/700test-ipsn/hypothesis', 'data/701test-ipsn/hypothesis', 'data/702test-ipsn/hypothesis', 'data/703test-ipsn/hypothesis', 'data/704test-ipsn/hypothesis']
            yolocust_def     = '../PyTorch-YOLOv3/config/yolov3-custom.cfg'
            yolocust_weights = '../PyTorch-YOLOv3/checkpoints_logdistance/yolov3_ckpt_5.pth'  # to be updated
            translate_net    = 'model/model1-12.13-net5-norm-32-splat.pt'                     # to be updated
            # below are useless
            max_ntx          = 10
            yolo_def         = '../PyTorch-YOLOv3/config/yolov3-custom-class.cfg'
            yolo_weights     = '../PyTorch-YOLOv3/checkpoints_logdistance_class/yolov3_ckpt_5.pth'
            dtxf_cnn1        =   'model_dtxf/12.13-cnn1-splat.pt'
            dtxf_cnn2_template = 'model_dtxf/12.13-cnn2-splat_{}.pt'
            return cls(max_ntx, test_data, train_data, ipsn_cov_list, ipsn_sen_list, ipsn_hypo_list, \
                       translate_net, yolocust_def, yolocust_weights, yolo_def, yolo_weights, dtxf_cnn1, dtxf_cnn2_template)

class IOUtility:
    '''input/output utility'''
    @staticmethod
    def read_logs(logs):
        '''reading logs
        Args:
            logs -- list<str> -- a list of filenames
        Return:
            data -- list<(Input, dic{str:Output}>
        '''
        data = []
        for log in logs:
            f = open(log, 'r')
            while True:
                line = f.readline()
                if line == '':
                    break
                myinput = Input.from_json_str(line)
                output_by_method = {}
                line = f.readline()
                while line != '' and line != '\n':
                    output = Output.from_json_str(line)
                    output_by_method[output.method] = output
                    line = f.readline()
                data.append((myinput, output_by_method))
        return data
