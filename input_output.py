'''
Encapsulate the input, output variables, as well as some default configurations
'''
from dataclasses import dataclass
from typing import List


@dataclass
class Default:
    '''some default constants
    '''
    alpha        = 3     # the slope of wireless signal depreciation
    std          = 1     # the standard deviation of the zero mean shadowing
    data_source  = 'propagation_model'
    methods      = ['dl']
    sen_density  = 100
    grid_length  = 100
    cell_length  = 10
    num_intruder = 1
    random_seed  = 0


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
    error: List
    false_alarm: float
    miss: float
    power: List
    preds: List
