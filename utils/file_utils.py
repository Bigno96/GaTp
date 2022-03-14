"""
Utility functions for dumping and retrieving data into/from files
"""

import logging
import os
import pickle

import numpy as np

from PIL import Image
from PIL.ImageOps import invert, colorize
from easydict import EasyDict
from typing import Any, List, Tuple


def create_dirs(dirs: List[str]) -> None:
    """
    Create directories in the system if not found
    :param: directory paths to create
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger('Dirs Creator').info(f'Creating directories error: {err}')
        exit(-1)


def save_image(file_path: str,
               input_map: np.array,
               start_pos_list: List[Tuple[int, int]]
               ) -> None:
    """
    Save image of the map (black and white) with agent starting positions (red)
    :param file_path: path to the specific file in the dataset directory
    :param input_map: shape=(H, W), matrix of '0' and '1'
    :param start_pos_list: starting positions coordinates over the map
    """
    img_path = f'{file_path}.png'
    # input map with float values
    f_map = np.array(input_map, dtype=float)
    for pos in start_pos_list:
        f_map[pos] = 0.5
    # white -> background | black -> obstacles | red -> agents starting positions
    colorize(invert(Image.fromarray(obj=np.uint8(f_map * 255))),
             black='black', white='white', mid='red').resize((256, 256),
                                                             resample=Image.BOX).save(img_path)


def dump_data(file_path: str,
              data: Any
              ) -> None:
    """
    Write date with pickle into specified file
    If the file already exists, overwrites it, else create it
    :param file_path: file to dump data into
    :param data: what to write into the file
                 has to be serializable with pickle
    """
    with open(file_path, 'wb') as f:
        pickle.dump(obj=data,
                    file=f,
                    protocol=pickle.HIGHEST_PROTOCOL)


def get_all_files(directory: str) -> List[str]:
    """
    Collect and return the list of all files in a directory and its subdirectories
    :param directory: path to the dataset directory
    :return: list of file_path
    """
    # return all file paths in the directory and its subdirectories
    return [os.path.join(dir_path, filename)
            for (dir_path, subdirs, files) in os.walk(directory)  # os.walk returns root, sub-dir list and file list
            for filename in files]  # get all filenames


def load_basename_list(data_path: str,
                       mode: str
                       ) -> List[str]:
    """
    Load a file basename list
    File basename -> 'mapID_caseID'
    :param data_path: path to the base dataset folder
    :param mode: 'train', 'valid', 'test', select mode dataset folder
    :return: list of str
    """
    # test, train or valid folder
    data_dir = os.path.join(data_path, mode)
    # get all and only filenames
    (_, _, filenames) = next(os.walk(data_dir))

    # filter out 'sol', 'png' and 'data' -> only basename
    return [name
            for name in filenames
            if 'sol' not in name
            and 'png' not in name
            and 'data' not in name]


class FolderSwitch:
    """
    Create train, valid and test folders
    Simulate a switch case with ranges to use during dataset splitting
    """

    def __init__(self,
                 dataset_dir: str,
                 config: EasyDict):
        """
        :param dataset_dir: directory of the dataset where to set up train, valid and test folders
        :param config: Namespace of dataset creation configurations
        """
        train_dir = os.path.join(dataset_dir, 'train')
        valid_dir = os.path.join(dataset_dir, 'valid')
        test_dir = os.path.join(dataset_dir, 'test')

        create_dirs([train_dir, valid_dir, test_dir])  # if not there, create

        # set how to divide scenarios between train, validation and test
        self.scenario_number = config.scenario_number   # save it for 'get_folder' method
        tot_scenario_number = config.map_number * self.scenario_number
        train_split = int(config.train_split * tot_scenario_number)
        valid_split = int(config.valid_split * tot_scenario_number)

        # dictionary to emulate a switch case with ranges
        self.folder_switch = {
            range(0, train_split): train_dir,
            range(train_split, train_split + valid_split): valid_dir,
            range(train_split + valid_split, tot_scenario_number): test_dir
        }

    def get_folder(self,
                   map_id: int,
                   sc_id: int
                   ) -> str:
        """
        Simulate a switch-case to decide which folder to use when splitting into train, valid and test
        :param map_id: id of the map on which scenarios are being created
        :param sc_id: id of the scenario to create
        :return: selected folder
        """
        key_val = int(map_id * self.scenario_number) + sc_id
        for key_range, value in self.folder_switch.items():
            if key_val in key_range:
                return value

        raise ValueError('Map ID out of switcher range')
