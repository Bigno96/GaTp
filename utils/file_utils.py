"""
Utility functions for dumping and retrieving data into/from files
"""

import logging
import os
import pickle

import numpy as np
from PIL import Image
from PIL.ImageOps import invert, colorize


def create_dirs(dirs):
    """
    Create directories in the system if not found
    :param: a list of directories to create
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger('Dirs Creator').info(f'Creating directories error: {err}')
        exit(-1)


def save_image(file_path, input_map, start_pos_list):
    """
    Save image of the map (black and white) with agent starting positions (red)
    :param file_path: path to the specific file in the dataset directory
    :param input_map: np.ndarray, size: H*W, matrix of '0' and '1'
    :param start_pos_list: list of tuples, (x,y) -> coordinates over the map
    """
    img_path = f'{file_path}.png'
    # input map with float values
    f_map = np.array(input_map, dtype=float)
    for pos in start_pos_list:
        f_map[pos] = 0.5
    # white -> background | black -> obstacles | red -> agents starting positions
    colorize(invert(Image.fromarray(obj=np.uint8(f_map * 255))),
             black='black', white='white', mid='red').resize((256, 256), resample=Image.BOX).save(img_path)


def create_folder_switch(dataset_dir, config):
    """
    Create train, valid and test folders
    Return a dictionary to emulate a switch case with ranges to use during dataset splitting
    :param dataset_dir: directory of the dataset where to set up train, valid and test folders
    :param config: Namespace of dataset creation configurations
    :return: dict, representing the switch-case
    """
    train_dir = os.path.join(dataset_dir, 'train')
    valid_dir = os.path.join(dataset_dir, 'valid')
    test_dir = os.path.join(dataset_dir, 'test')

    create_dirs([train_dir, valid_dir, test_dir])  # if not there, create

    # set how to divide scenarios between train, validation and test
    train_split = int(config.train_split * config.map_number)
    valid_split = int(config.valid_split * config.map_number)

    # dictionary to emulate a switch case with ranges
    folder_switcher = {
        range(0, train_split): train_dir,
        range(train_split, train_split + valid_split): valid_dir,
        range(train_split + valid_split, config.map_number): test_dir
    }

    return folder_switcher


def get_folder_from_switch(map_id, switcher_dict):
    """
    Simulate a switch-case to decide which folder to use when splitting into train, valid and test
    :param map_id: id of the map on which scenarios are being created
    :param switcher_dict: dict -> {range : folder}
    :return: selected folder
    """
    for key, value in switcher_dict.items():
        if map_id in key:
            return value

    raise ValueError('Map ID out of switcher range')


def dump_data(file_path, data):
    """
    Write date with pickle into specified file
    If the file already exists, overwrites it, else create it
    :param file_path: file to dump data into
    :param data: what to write into the file
                 has to be serializable with pickle
    """
    with open(file_path, 'wb') as f:
        pickle.dump(obj=data, file=f,
                    protocol=pickle.HIGHEST_PROTOCOL)


def get_all_files(directory):
    """
    Collect and return the list of all files in a directory and its subdirectories
    :param directory: path to the dataset directory
    :return: list of file_path
    """
    # return all file paths in the directory and its subdirectories
    return [os.path.join(dir_path, filename)
            for (dir_path, subdirs, files) in os.walk(directory)  # os.walk returns root, sub-dir list and file list
            for filename in files]  # get all filenames
