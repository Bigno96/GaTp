from data_loading.dataset import GaTpDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from pprint import pprint


def display(dataset_path):

    train_dataset = GaTpDataset(data_dir=dataset_path, mode='train', expert_type='tp')
    valid_dataset = GaTpDataset(data_dir=dataset_path, mode='valid', expert_type='tp')
    test_dataset = GaTpDataset(data_dir=dataset_path, mode='test', expert_type='tp')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    print('Showing train dataset')
    iterate_data(dataloader=train_dataloader)

    print('Showing validation dataset')
    iterate_data(dataloader=valid_dataloader)

    print('Showing test dataset')
    iterate_data(dataloader=test_dataloader)


def iterate_data(dataloader):

    it = iter(dataloader)
    end = False

    while not end:
        image, environment, expert_sol = next(it)

        plt.imshow(image.squeeze())
        plt.show()
        pprint(environment, sort_dicts=False)
        expert_sol['schedule'] = torch.transpose(expert_sol['schedule'].squeeze(), 0, 2)    # print formatting
        pprint(expert_sol, sort_dicts=False)

        input('\nPress Enter to continue\n')


if __name__ == '__main__':
    __spec__ = None
    torch.set_printoptions(profile='full')

    data_path = 'D:\\Uni\\TESI\\GaTp\\datasets\\random_grid\\20x20map\\0.1density\\' \
                   '20agents_500tasks_0split_+1_every1\\random_start+avoid_non_task_rep_task'
    display(dataset_path=data_path)
