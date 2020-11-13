'''
Encapsulate the input, output variables, as well as some default configurations
'''
from dataclasses import dataclass
from typing import List
import json


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
            'num_intruder':self.num_intruder,
            'experiment_num':self.experiment_num,
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
        myinput.num_intruder   = json_dict['num_intruder']
        myinput.experiment_num = json_dict['experiment_num']
        myinput.image_index    = json_dict['image_index']
        myinput.methods        = json_dict['methods']
        myinput.data_source    = json_dict['data_source']
        myinput.sensor_density = json_dict['sensor_density']
        return myinput

    def log(self):
        '''log'''
        return self.to_json_str()


@dataclass
class Output:
    '''encapsulate the output variables
    '''
    method: List[str]
    error: List            # error of the detected TX
    false_alarm: int
    miss: int
    preds: List
    time: float


    def to_json_str(self):
        '''return json formated string
        Return:
            str
        '''
        outputdict = {
            "method":self.method,
            "error":self.error,
            "false_alarm":self.false_alarm,
            "miss":self.miss,
            "preds":self.preds,
            "time":self.time
        }
        return outputdict

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
