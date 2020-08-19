'''
Input and output representation of the deep learning networks
'''

class InputRepresentation:
    '''Convert the sensor's data into a 2D matrix
    '''
    def __init__(self, sensing_raw):
        self.sensing_raw = sensing_raw
        self.sensing_image = None

    def transform2image(self):
        '''transform the raw sensing data into a image that can be utilized by deep learning frameworks
        '''
        pass 
