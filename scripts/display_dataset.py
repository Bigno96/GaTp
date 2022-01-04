import os
import pickle
from pprint import pprint

from PIL import Image


def display():
    # change this to change which dataset to read
    dataset_path = "D:\\Uni\\TESI\\GaTp\\datasets\\random_grid\\20x20map\\0.2density\\10agents_50tasks\\random_start" \
                   "+avoid_non_task_rep_task"

    for dir_name in os.listdir(dataset_path):
        print(f'Showing {dir_name} folder')
        dir_path = os.path.join(dataset_path, dir_name)

        file_set = {f for (r, d_l, f_l) in os.walk(dir_path)
                     for f in f_l}
        img_set = {f for f in file_set
                    if f.endswith('.png')}
        exp_set = {f for f in file_set
                   if 'sol' in f}
        env_set = file_set - img_set - exp_set

        for (img, env, exp) in zip(img_set, env_set, exp_set):
            img_path = os.path.join(dir_path, img)
            env_path = os.path.join(dir_path, env)
            exp_path = os.path.join(dir_path, exp)

            im = Image.open(img_path)
            im.show(title=img)
            with open(env_path, 'rb') as _d1:
                pprint(pickle.load(_d1))
            with open(exp_path, 'rb') as _d2:
                pprint(pickle.load(_d2))

            input("\nPress Enter to continue...\n")


if __name__ == '__main__':
    __spec__ = None
    display()
