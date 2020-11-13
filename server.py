'''Localization server
'''

import os
import time
import argparse
import torch
import sys
import numpy as np
from flask import Flask, request
from mydnn import NetTranslation, NetNumTx
import mydnn_util
from input_output import Input, Output, Default

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world!'

@app.route('/localize', methods=['POST'])
def localize():
    '''localize
    '''
    myinput = Input.from_json_dict(request.get_json())

    if data_source == None:
        data_source = myinput.data_source
        sensor_input_dataset = mydnn_util.SensorInputDatasetTranslation(root_dir=data_source, transform=mydnn_util.tf)
    else:
        if data_source != myinput.data_source:
            raise Exception('data source incnosistant {} and {}'.foramt(data_source, myinput.data_source))

    outputs = []
    if 'dl' in myinput.methods:
        sample = sensor_input_dataset[myinput.image_index]
        X      = torch.as_tensor(sample['matrix']).unsqeeze(0).to(device)
        y_f    = np.expand_dims(sample['target_float'], 0)
        indx   = np.expand_dims(np.array(sample['index']), 0)
        start = time.time()
        pred_matrix = model1(X)
        pred_ntx    = model2(X)
        pred_matrix = pred_matrix.data.cpu().numpy()
        _, pred_ntx = pred_ntx.data.cpu().max(1)
        pred_ntx = (pred_ntx + 1).numpy()
        preds, errors, misse, false = mydnn_util.Metrics.localization_error_image_continuous(pred_matrix.copy(), pred_ntx, y_f, indx, Default.grid_length, peak_threshold=1, debug=True)
        end = time.time()
        outputs.append(Output('dl', errors[0], false[0], misse[0], preds[0], end-start))
    if 'map' in myinput.methods:
        pass
    if 'splot' in myinput.methods:
        pass
    if 'dtxf' in myinput.methods:
        pass

    


    return 'hello world'


if __name__ == '__main__':

    hint = 'python server.py -maxn 5'
    parser = argparse.ArgumentParser(description='server side. | hint')

    max_ntx = 5
    path1 = 'model/model1-11.12.pt'
    path2 = 'model/model2-11.12-2.pt'
    device = torch.device('cuda')
    model1 = NetTranslation()
    model1.load_state_dict(torch.load(path1))
    model1 = model1.to(device)
    model2 = NetNumTx(max_ntx)
    model2.load_state_dict(torch.load(path2))
    model2.to(device)
    model2 = model2.to(device)
    model1.eval()
    model2.eval()


    data_source = None

    app.run(host="0.0.0.0", port=5000)
