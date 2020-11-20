'''
common utility
'''

import torch
import math
import os
import shutil
import time
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from torch._six import container_abcs, string_classes, int_classes
import re
np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


class Utility:
    '''some common utilities'''

    @staticmethod
    def distance_propagation(indx2d_1: tuple, indx2d_2: tuple):
        '''euclidean distance for propagation model'''
        if indx2d_1 == indx2d_2:
            return 0.5
        else:
            return math.sqrt((indx2d_1[0] - indx2d_2[0]) ** 2 + (indx2d_1[1] - indx2d_2[1]) ** 2)

    @staticmethod
    def distance(indx2d_1: tuple, indx2d_2: tuple):
        '''euclidean distance for localization error'''
        return math.sqrt((indx2d_1[0] - indx2d_2[0]) ** 2 + (indx2d_1[1] - indx2d_2[1]) ** 2)

    @staticmethod
    def remove_make(root_dir: str):
        '''if root_dir exists, remove all the content
           make directory
        '''
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
        os.mkdir(root_dir)

    @staticmethod
    def db2linear(db: float):
        '''Transform power from decibel into linear format
        Args:
            db -- power in db
        Return:
            float -- power in linear form
        '''
        try:
            if db <=-80:
                return 0
            linear = np.power(10, db/10)
            return linear
        except Exception as e:
            print(e, db)

    @staticmethod
    def linear2db(linear: float):
        '''Transform power from linear into decibel format
        Args:
            linear -- power in linear
        Return:
            float -- power in decibel
        '''
        db = 10 * np.log10(linear)
        if db < -80 or db is np.nan:
            db = -80
        return db

    @staticmethod
    def detect_peak(image, num_tx: int, threshold=0.05):  # TUNE: a larger threshold will decrease false
        """
        Returns a boolean mask of the peaks (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        Args:
            image  -- array-like -- (100,100)
            num_tx -- int -- number of Tx
            threshold -- float -- threshold for non-tx areas
        Return:
            list<(int, int)>: a list of peaks
        """
        def detect_helper(size):
            '''Takes an image and detect the peaks using the local maximum filter
            '''
            if size in memo:
                return memo[size]

            threshold_mask = image < threshold
            image[threshold_mask] = 0
            neighborhood = np.array([[True for _ in range(size)] for _ in range((size))])
            #apply the local maximum filter; all pixel of maximal value
            #in their neighborhood are set to 1
            local_max = maximum_filter(image, footprint=neighborhood)==image
            #local_max is a mask that contains the peaks we are
            #looking for, but also the background.
            #In order to isolate the peaks we must remove the background from the mask.
            #we create the mask of the background
            background = (image < threshold)
            #a little technicality: we must erode the background in order to
            #successfully subtract it form local_max, otherwise a line will
            #appear along the background border (artifact of the local maximum filter)
            eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
            #we obtain the final mask, containing only peaks,
            #by removing the background from the local_max mask (xor operation)
            detected_peaks = local_max ^ eroded_background

            memo[size] = detected_peaks
            return detected_peaks

        memo = {}
        size = [40, 30, 20, 15, 10, 5]             # TUNE: a larger size will decrease false
        peaks = []
        peaks_num = []
        for i, s in enumerate(size):               # first pass with coarse grain size
            peaks = detect_helper(s)
            peaks = np.where(peaks == True)
            peaks_num.append(len(peaks[0]))
            if i == 0 and peaks_num[0] > num_tx:
                return [(x, y) for x, y in zip(peaks[0], peaks[1])], s
            if len(peaks[0]) == num_tx:
                return [(x, y) for x, y in zip(peaks[0], peaks[1])], s

        new_size = []            # second pass with fine coarse size
        for i in range(len(peaks_num)-1):
            if peaks_num[i] < num_tx < peaks_num[i+1]:
                new_size = list(range(size[i], size[i+1]-1, -1))
                break
        else:
            new_size = [5, 4, 3, 2]
        peaks_num = []
        for s in new_size:
            peaks = detect_helper(s)
            peaks = np.where(peaks == True)
            peaks_num.append(len(peaks[0]))
            if len(peaks[0]) == num_tx:
                return [(x, y) for x, y in zip(peaks[0], peaks[1])], s

        min_diff = 100
        min_i = 0
        for i in range(len(peaks_num)):
            diff = abs(num_tx - peaks_num[i])
            if diff <= min_diff:
                min_diff = diff
                min_i = i
        peaks = detect_helper(new_size[min_i])
        peaks = np.where(peaks == True)
        return [(x, y) for x, y in zip(peaks[0], peaks[1])], new_size[min_i]


    @staticmethod
    def compute_error(pred_locations, true_locations, distance_threshold, debug=False):
        '''Given the true location and localization location, computer the error **for our localization**
        Args:
            true_locations (list): an element is a tuple (true transmitter 2D location)
            true_powers (list):    an element is a float
            distance_threshold -- float -- a distance threshold for a predicted location being valid
        Return:
            (tuple): (list, float, float, list), (distance error, miss, false alarm, power error)
        '''
        if len(pred_locations) == 0:
            return [], 1, 0
        distances = np.zeros((len(true_locations), len(pred_locations)))
        for i in range(len(true_locations)):
            for j in range(len(pred_locations)):
                distances[i, j] = np.sqrt((true_locations[i][0] - pred_locations[j][0]) ** 2 + (true_locations[i][1] - pred_locations[j][1]) ** 2)

        k = 0
        matches = []
        misses = list(range(len(true_locations)))
        falses = list(range(len(pred_locations)))
        while k < min(len(true_locations), len(pred_locations)):  # minimum weight matching
            min_error = np.min(distances)
            min_error_index = np.argmin(distances)
            i = min_error_index // len(pred_locations)
            j = min_error_index %  len(pred_locations)
            matches.append((i, j, min_error))
            distances[i, :] = np.inf
            distances[:, j] = np.inf
            k += 1

        errors = []               # distance error
        detected = 0
        for match in matches:
            error = match[2]
            if error <= distance_threshold:
                errors.append(round(error, 4))
                detected += 1
                misses.remove(match[0])
                falses.remove(match[1])

        if debug:
            print('\nPred:', end=' ')
            for match in matches:
                print(str(pred_locations[match[1]]).ljust(9), end='; ')
            print('\nTrue:', end=' ')
            for match in matches:
                print(str(true_locations[match[0]]).ljust(9), end='; ')
            print('\nMiss:', end=' ')
            for miss in misses:
                print(true_locations[miss], end=';  ')
            print('\nFalse Alarm:', end=' ')
            for false in falses:
                print(pred_locations[false], end=';  ')
            print()
        try:
            return errors, len(true_locations) - detected, len(pred_locations) - detected  # error, miss (FN), false (FP)
        except:
            return [], 0, 0


if __name__ == '__main__':
    pred_image = np.loadtxt('test.txt')
    # print(pred_image[1, 3])
    peaks = Utility.detect_peak(pred_image, 5, 1)
    print(peaks)


# NOTE: there are some cases where predicting num of TX is wrong, but the output peaks are right
