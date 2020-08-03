'''

The main:
1. generate synthetic sensing data
2. transform sensing data into images
3. use a deep learning model to process the images
4. get results

'''

import argparse
from input_output import Default
from generate import GenerateData


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Localize multiple transmitters')

    parser.add_argument('-gd', '--generate_data', action='store_true')
    parser.add_argument('-al', '--alpha', nargs=1, type=float, default=[Default.alpha], help='the slope of pathloss')
    parser.add_argument('-st', '--std', nargs=1, type=float, default=[Default.std], help='the standard deviation zero mean Guassian')
    parser.add_argument('-gl', '--grid_length', nargs=1, type=int, default=[Default.grid_length], help='the length of the grid')
    parser.add_argument('-cl', '--cell_length', nargs=1, type=float, default=[Default.cell_length], help='the length of a cell')
    parser.add_argument('-rs', '--random_seed', nargs=1, type=int, default=[Default.random_seed], help='random seed')
    parser.add_argument('-sd', '--sensor_density', nargs=1, type=int, default=[Default.sen_density], help='number of sensors')

    args = parser.parse_args()

    if args.generate_data:
        print('generating data')
        random_seed = args.random_seed[0]
        alpha = args.alpha[0]
        std = args.std[0]
        grid_length = args.grid_length[0]
        cell_length = args.cell_length[0]
        sensor_density = args.sensor_density[0]

        print(random_seed, alpha, std, grid_length, cell_length, sensor_density)
        generatedata = GenerateData(random_seed, alpha, std, grid_length, cell_length, sensor_density)
