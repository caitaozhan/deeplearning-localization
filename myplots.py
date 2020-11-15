import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_sensor_output(image, true_locations):
    '''Visualize the intruders and sensor_output to have a better sense on deciding the threshold
    Args:
        image -- np.ndarray, n=3, eg. (1, 100, 100)
        true_locations -- list<(float, float>)
    '''
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    sns.heatmap(image[0], cmap='gray', ax=ax)
    plt.title('Intruders: ' + ' '.join(map(lambda intru: '({:2d}, {:2d})'.format(int(intru[0]), int(intru[1])), true_locations)), fontsize=20)
    # guarantee_dir('visualize/localization')
    plt.savefig('visualize/localization/{}-sensor-output'.format(fig))
