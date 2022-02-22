"""
Class for converting environment data and expert solutions into neural network compatible data
This conversion is always done at dataset creation

### See GaTp/utils/transform_data.py for more information about data transformation ###

'FOV' and 'comm_radius' parameters used here are taken from 'GaTp/yaml_configs/dataset_creation.yaml'

If you wish to feed the ML model data with different FOV or Communication Radius,
when launching training or testing use the FLAG '-transform_runtime_data' and specify with '-FOV' and/or '-comm_radius'
the new values
In this way, NN input data will be generated at Data Loading time


N.B.: Generating input data at 'runtime' takes a bit of time

### Be sure to feed the model with FOV and Comm Radius values equal to the one used for its training ###
"""
import os

from p_tqdm import p_map

from utils.transform_data import DataTransformer
from utils.file_utils import load_basename_list, dump_data


def get_nn_data(config, dataset_dir, bad_instances_list=(), recovery_mode=False, file_path_list=None):
    """
    Get NN compatible data out of environment and expert solution files
    :param config: Namespace of dataset configurations
    :param dataset_dir: path to the dataset directory
    :param bad_instances_list, list with the file path of bad MAPD instance files
    :param recovery_mode: boolean, True if Get_nn_data is used to re-compute bad MAPD instances
    :param file_path_list: list of file path containing environment data to run expert over
                           Pass this ONLY with recovery_mode = True
    """

    mode_list = ['train', 'valid', 'test']

    # get bad instances basename (mapID_caseID)
    bad_instances_basename = {os.path.basename(os.path.normpath(file_path))
                              for file_path in bad_instances_list
                              }

    # for each folder in the dataset
    for mode in mode_list:

        # regenerating bad instances of MAPD
        if recovery_mode:
            # no file paths given while recovery mode
            if file_path_list is None:
                raise ValueError('Experts launched in recovery mode with no file paths')

            basename_list = {os.path.basename(os.path.normpath(file_path))
                             for file_path in file_path_list
                             if mode in file_path             # only 'mode' folder
                             }

            # discard bad instances, even in recovery mode, since no expert solution for them
            basename_list = list(basename_list - bad_instances_basename)

        # first generation
        else:
            basename_list = load_basename_list(data_path=dataset_dir, mode=mode)
            basename_list = list(set(basename_list) - bad_instances_basename)      # discard bad instances

        # if there are files to transform
        if basename_list:
            # launch multiprocessing data transformation
            worker = DataTransformerWorker(config=config, mode=mode, dataset_dir=dataset_dir)
            print(f'Transforming {mode} data')
            p_map(worker, basename_list)


class DataTransformerWorker:
    """
    Class for multiprocess data transformation
    """

    def __init__(self, config, mode, dataset_dir):
        """
        :param config: Namespace of dataset configurations
        :param mode: 'train', 'valid', 'test'
        :param dataset_dir: path to the dataset directory
        """
        self.config = config
        self.mode = mode
        self.dataset_dir = dataset_dir

        self.data_transform = DataTransformer(config=config,
                                              data_path=dataset_dir,
                                              mode=mode)

    def __call__(self, basename):
        """
        :param basename: file basename (mapID_caseID)
        """
        # apply data transformation
        data = self.data_transform.get_data(basename=basename)
        # get complete file path for dumping
        file_path = os.path.join(self.dataset_dir, self.mode, f'{basename}_data')
        # data are saved as they come from data_transform.get_data
        dump_data(file_path=file_path, data=data)
