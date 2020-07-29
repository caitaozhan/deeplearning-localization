'''
Encapsulates both RX and TX
'''

class Node:
    '''the parent class of RX and TX
    '''
    def __init__(self, x: int, y: int, indx: int):
        '''
        Args:
            x -- x axis coordinate
            y -- y axis coordinate
            indx -- index
        '''
        self.x = x
        self.y = y
        self.indx = indx


class Receiver(Node):
    '''encapsualte a receiver
    '''
    def __init__(self, x: int, y: int, indx: int):
        '''init'''
        super().__init__(x, y, indx)


class Transmitter(Node):
    '''encapsulate a transmitter
    '''
    def __init__(self, x: int, y: int, indx: int, power: int):
        super().__init__(x, y, indx)
        self.power = power

