'''
Encapsulate the input and output variables
'''
from dataclasses import dataclass


@dataclass
class Default:
    alpha: float == 2   # the slope of wireless signal depreciation
    


@dataclass
class Input:
    length: int
    width: int
    