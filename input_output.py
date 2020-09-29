'''
Encapsulate the input, output variables, as well as some default configurations
'''
from dataclasses import dataclass
from typing import List


@dataclass
class Default:
    '''some default configurations
    '''
    alpha            = 3.5     # the slope of wireless signal depreciation
    std              = 1       # the standard deviation of the zero mean shadowing
    data_source      = 'propagation_model'
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

@dataclass
class Input:
    '''encapsulate the input variables
    '''
    num_intruder: int = Default.num_intruder
    data_source: str  = Default.data_source


@dataclass
class Output:
    '''encapsulate the output variables
    '''
    method: str
    num_tx: int
    num_detected: int
    error: List            # error of the detected TX
    false_alarm: int
    miss: int
    preds: List

    # def get_avg_error(self):
    #     '''the average error
    #     '''
    #     return np.mean(self.error)
    
    # def get_method(self):
    #     return self.method

    # def get_num_tx(self):
    #     return self.num_tx
    
    # def get_detected(self):
    #     return self.num_detected
    
    # def get_error(self):
    #     return self.error
    
    # def get_false_alarm(self):
    #     return self.false_alarm
    
    # def get_miss(self):
    #     return self.miss
    
    # def preds(self):
    #     return self.preds
    
    # def to_json_str(self):
    #     '''return json formated string
    #     Return:
    #         str
    #     '''
    #     outputdict = {
    #         "method":self.method,
    #         "num_tx":self.num_tx,
    #         "error":self.error,
    #         "false_alarm":self.false_alarm,
    #         "miss":self.miss,
    #         "power":self.power,
    #         "time":self.time,
    #     }
    #     return outputdict