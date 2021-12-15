import logging
import os


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
