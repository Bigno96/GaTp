from PIL import Image
import pickle
import os
from pprint import pprint


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
        data_set = file_set - img_set

        for (img, data) in zip(img_set, data_set):
            img_path = os.path.join(dir_path, img)
            data_path = os.path.join(dir_path, data)
            im = Image.open(img_path)
            im.show(title=img)
            with open(data_path, 'rb') as d:
                pprint(pickle.load(d))

            input("Press Enter to continue...\n")


if __name__ == '__main__':
    __spec__ = None
    display()
