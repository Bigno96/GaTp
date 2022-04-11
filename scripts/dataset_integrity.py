import torch
import time

from easydict import EasyDict
from tqdm import tqdm

import data_loading.data_loader as loader
from models.basic.mlp import MLP


def check_integrity():

    # put manually values into config Namespace
    config = EasyDict()
    config.batch_size = 10
    config.valid_batch_size = 1
    config.test_batch_size = 1
    config.data_loader_workers = 0
    config.pin_memory = True
    config.data_root = 'D:/Uni/TESI/GaTp/datasets'
    config.map_type = 'random_grid'
    config.map_shape = [20, 20]
    config.map_density = 0.1
    config.agent_number = 20
    config.task_number = 500
    config.imm_task_split = 0.0
    config.new_task_per_timestep = 1
    config.step_between_insertion = 1
    config.start_position_mode = 'random'
    config.task_creation_mode = 'avoid_non_task_rep'
    config.transform_runtime_data = False
    config.expert_type = 'tp'
    config.FOV = 9
    config.shuffle_train = False

    print(f'Loading Data')
    time.sleep(.1)  # printing sync

    config.mode = 'train'
    train_dl = loader.GaTpDataLoader(config=config)
    config.mode = 'test'
    test_dl = loader.GaTpDataLoader(config=config)

    print(f'Checking Train Dataset')
    time.sleep(.1)  # printing sync

    # toy model
    model = MLP(in_features=int(3 * 11 * 11),
                out_features=5,
                hidden_features=(5000, 500))

    '''training dataset test'''
    # batch input shape = batch_size, num_agent, 3, fov_h, fov_w
    # batch gso shape = batch_size, num_agent, num_agent
    # batch target shape = batch_size, num_agent, 5
    for batch_input, batch_GSO, batch_target, basename \
            in tqdm(train_dl.train_loader):

        predict = model(batch_input.reshape(config.batch_size * config.agent_number, 3 * 11 * 11))
        print(f'predict: {predict}')

        batch_target = batch_target.reshape(config.batch_size * config.agent_number, 5)
        print(f'max batch target: {torch.max(batch_target, 1)[1]}')

        loss = torch.nn.CrossEntropyLoss(predict, torch.max(batch_target, 1)[1])
        print(f'loss {loss}')

        # check_corruption(batch_input, basename)
        # check_corruption(batch_GSO, basename)
        # check_corruption(batch_target, basename)

        input('Oscar è scemoh\n')

    time.sleep(.1)  # printing sync
    print(f'Checking Valid Dataset')
    time.sleep(.1)  # printing sync

    '''valid dataset test'''
    # obstacle map shape = batch_size, num_agent, num_agent
    # start pos list shape = batch_size, num_agent, 2
    # task list shape = batch_size, num_task, 2, 2
    # makespan shape = 1
    # service time shape = 1
    for obstacle_map, start_pos_list, task_list, makespan, service_time, basename \
            in tqdm(train_dl.valid_loader):

        check_corruption(obstacle_map, basename)
        check_corruption(start_pos_list, basename)
        check_corruption(task_list, basename)
        check_corruption(makespan, basename)
        check_corruption(service_time, basename)

    time.sleep(.1)  # printing sync
    print(f'Checking Test Dataset')
    time.sleep(.1)  # printing sync

    # test dataset test
    for obstacle_map, start_pos_list, task_list, makespan, service_time, basename \
            in tqdm(test_dl.test_loader):

        check_corruption(obstacle_map, basename)
        check_corruption(start_pos_list, basename)
        check_corruption(task_list, basename)
        check_corruption(makespan, basename)
        check_corruption(service_time, basename)


def check_corruption(t, basename):
    if not torch.all(torch.isfinite(t)).tolist():
        print(f'Corrupted input in {basename}')


if __name__ == '__main__':
    __spec__ = None
    check_integrity()
