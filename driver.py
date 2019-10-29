import warnings
warnings.filterwarnings('ignore')

import sys
import os
my_path = os.path.dirname(os.path.realpath(__file__))

import random
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate
from keras.datasets import mnist
# from keras.utils import np_utils

# Import MNIST data from keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Check the downloaded data
print("Shape of training dataset: {}".format(np.shape(X_train)))
print("Shape of test dataset: {}".format(np.shape(X_test)))
print("Label for the image: {}".format(y_train[0]))

# Make the index dictionary for each number
label_dic = collections.defaultdict(list)
for i, d in enumerate(y_train):
    label_dic[d].append(i)

# Check the index dictionary
counter = 0
for k in sorted(label_dic.keys()):
    print("Digit: {}, #: {}".format(k, len(label_dic[k])))
    counter += len(label_dic[k])
print("Total #: {}".format(counter))

# Function for image rotation, the default rotation angle in degrees is set from 0 to 30
def image_rotation(image, angle_range=(0, 30)):
    h, w = image.shape
    angle = np.random.randint(*angle_range)
    image = rotate(image, angle)
    image = imresize(image, (h, w))
    return image

# Function for image obstruction, the size usually set as the 1/3 of the image size
def image_obstruction(image, mask_size):
    mask_value = image.mean()

    h, w = image.shape
    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size
    if top < 0:
        top = 0
    if left < 0:
        left = 0
    
    image[top: bottom, left: right].fill(mask_value)
    return image

# Function for the sequence generation
def generate_numbers_sequence(digits, min_spacing = 0, max_spacing = 0, image_width = 28, \
                              image_height = 28, rotation = False, obstruction = False):
    # Randomly sampling index
    index = []
    for digit in digits:
        index.append(random.choice(label_dic[int(digit)]))
    
    # Make list of images
    images_list = []
    for i in index[:-1]:
        image = imresize(X_train[i], (int(image_height), int(image_width)))
        # Data augmentation: rotation
        if rotation:
            image = image_rotation(image)
        images_list.append(np.c_[image, np.zeros((int(image_height), random.randint(int(min_spacing), int(max_spacing))))])
    last_image = X_train[index[-1]]
    # Data augmentation: rotation
    if rotation:
        last_image = image_rotation(last_image)
    images_list.append(last_image)
    
    # Stitch images together
    output = np.hstack(images_list)
    
    # Data augmentation: obstruction
    if obstruction:
        output = image_obstruction(output, int(image_width) // 3)
    
    return output, digits


if __name__ == '__main__':

    # Input by command line, the parameters: digits(necessary), min_spacing, max_spacing, image_width
    digits = sys.argv[1]
    min_spacing = sys.argv[2] if len(sys.argv) >= 3 else 0
    max_spacing = sys.argv[3] if len(sys.argv) >= 4 else 0
    image_width = sys.argv[4] if len(sys.argv) >= 5 else 28
    result, label = generate_numbers_sequence(digits, min_spacing, max_spacing, image_width)

    plt.figure()
    plt.axis('off')
    plt.imshow(result, cmap='gray')
    image_name = ''.join([str(d) for d in digits])
    plt.savefig('output/' + image_name + '.png', bbox_inches = 'tight', pad_inches = 0)


    # # Input by excel file
    # given_numbers = pd.read_excel(my_path + '/input.xlsx', dtype=str)

    # for i, row in given_numbers.iterrows():
    #     digits = [s for s in filter(str.isdigit, row['digits'])]
    #     min_spacing = row['min_spacing']
    #     max_spacing = row['max_spacing']
    #     image_width = row['width']
    #     image_height = row['height']
    #     rotation = True if row['rotation'] == '1' else False
    #     obstruction = True if row['obstruction'] == '1' else False

    #     result, label = generate_numbers_sequence(digits, min_spacing, max_spacing, image_width, \
    #                                               image_height, rotation, obstruction)

    #     plt.figure()
    #     plt.axis('off')
    #     plt.imshow(result, cmap='gray')
    #     image_name = ''.join([str(d) for d in digits])
    #     image_name += '_rotate' if rotation else ''
    #     image_name += '_obstructe' if obstruction else ''
    #     plt.savefig('output2/' + str(i + 1) + '_' + image_name + '.png', bbox_inches = 'tight', pad_inches = 0)
