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
    sen_density      = 500
    grid_length      = 100
    cell_length      = 10
    num_intruder     = 1
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
    methods          = ['dl', 'map', 'splot', 'dtxf']
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
        outputdict = {
            "ground_truth": self.ground_truth,
            "sensor_data":  self.sensor_data
        }
        return json.dumps(outputdict)

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
        else:
            raise Exception('unknown metrics')

    def to_json_str(self):
        '''return json formated string
        Return:
            str
        '''
        self.preds = [(round(x, 2), round(y, 2)) for x, y in self.preds]
        self.time = round(self.time, 3)
        outputdict = {
            "method":self.method,
            "error":self.error,
            "false_alarm":self.false_alarm,
            "miss":self.miss,
            "preds":self.preds,
            "time":self.time
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
        return cls(method, error, false_alarm, miss, preds, time)

    def log(self):
        return self.to_json_str()


@dataclass
class DataInfo:
    '''the data set used for training and testing
    '''
    test_data: str
    train_data: str
    ipsn_cov: str
    ipsn_sensors: str
    ipsn_hypothesis: str
    dl_model1: str     # image translation
    dl_model2: str     # predict num of TX

    @classmethod
    def naive_factory(cls, data_source):
        '''factory'''
        if data_source == 'data/60test':
            test_data = 'data/60test'
            train_data = 'data/60train'
            ipsn_cov = 'data/60train-ipsn/cov'
            ipsn_sensors = 'data/60train-ipsn/sensors'
            ipsn_hypothesis = 'data/60train-ipsn/hypothesis'
            dl_model1 = 'model/model1-11.12.pt'
            dl_model2 = 'model/model2-11.12-2.pt'
            return cls(test_data, train_data, ipsn_cov, ipsn_sensors, ipsn_hypothesis, dl_model1, dl_model2)


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
