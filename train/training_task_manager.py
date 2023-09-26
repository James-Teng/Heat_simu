# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.3.8 15:35
# @Author  : James.T
# @File    : task_manager.py

import argparse
import csv
import time
import os
import json
from json.decoder import JSONDecodeError
import sys
from typing import Optional

config_template = {
    'create_time': 'None',
    'description': 'None',
    'is_config': False,  # a flag determining the task is configured or not

    'train_dataset_config': {
        'train_dataset_root': 'COCO2014',  # what dataset to use
        'crop_size': 96,  # crop
    },
    'generator_config': {
        'large_kernel_size_g': 9,  # 第一层卷积和最后一层卷积的核大小
        'small_kernel_size_g': 3,  # 中间层卷积的核大小
        'n_channels_g': 32,  # 中间层通道数
        'n_blocks_g': 2,  # 残差模块数量
    },
    'hyper_params': {
        'total_epochs': None,
        'batch_size': None,
        'lr_initial': None,
        'lr_decay_gamma': None,
        'lr_milestones': [],
    },
    'others': {
        'n_gpu': 1,
        'worker': 4,
    },
}
# 支持外部导入 config，也可以命令行输入

# 针对单个训练任务，参数配置，文件夹建立（中间结果，训练log，loss，模型文件夹checkpoint）
# 初始化时，传入一个路径，如果不存在，则创建一个完整的新任务，存在则记录路径，并读取配置，然后什么都不做
# 有三种创建新任务的方式
# 可以返回配置列表，可以返回具体文件夹路径

# 提供一些命名用的函数

class TaskManager:
    """
    provide functions to manage a task
    """
    def __init__(
            self,
            task_path: str,
            external_config_path: Optional[str] = None,
            cmd_config: Optional[list] = None,
    ):
        """
        initialization
        """
        self.task_path = task_path
        self.checkpoint_path = os.path.join(task_path, 'checkpoint')
        self.record_path = os.path.join(task_path, 'record')
        self.config_path = os.path.join(task_path, 'config.json')

        # new task
        if not os.path.exists(task_path):

            print('TaskManager: creating new task')

            # create path
            os.makedirs(self.task_path)
            os.mkdir(self.checkpoint_path)
            os.mkdir(self.record_path)

            # config
            if external_config_path:
                try:
                    self.config = self.__read_task_config(external_config_path)
                except FileNotFoundError:
                    print(f'TaskManager: no config file found at \'{external_config_path}\'')
                    sys.exit()
            elif cmd_config:
                self.config = cmd_config
            else:
                self.config = config_template

            try:
                self.__write_task_config(config_template, self.config_path)
            except FileExistsError:
                print(f'TaskManager: config \'{external_config_path}\' already exists')
                sys.exit()

        # task already exists
        else:
            print('TaskManager: reading existing task')
            try:
                self.config = self.__read_task_config(self.config_path)
            except FileNotFoundError:
                print(f'TaskManager: no config file found at \'{self.config_path}\'')
                sys.exit()


    def new_task(self, new_task_id):

        # create directory, you can change dir arragement here
        task_brief = {'create_time': time.strftime('%Y%m%d_%H%M_%A', time.localtime()),
                      'task_id': new_task_id}
        task_path = self.__get_task_path(task_brief)
        try:
            os.mkdir(task_path)
            os.mkdir(os.path.join(task_path, 'checkpoint'))
            os.mkdir(os.path.join(task_path, 'record'))
        except FileExistsError:
            print(f'TaskManager: {task_path} or sth in it already exist')
            sys.exit()

        # create config file
        config_file = os.path.join(task_path, f'{new_task_id}_config.json')
        new_task_config = task_config_template
        new_task_config['task_id'] = new_task_id
        new_task_config['create_time'] = time.strftime(f'%Y.%m.%d %H:%M:%S %A', time.localtime())
        try:
            self.__write_task_config(new_task_config, config_file)
        except FileExistsError:
            print(f'TaskManager: {config_file} already exist')
            sys.exit()

        # register new task
        self.__add_register_list(task_brief)

    def display_task_config(self, task_id):
        """
        display task_config of a task named task_id

        :param task_id: the task you want to show
        :returns: None
        """
        config = self.get_task_config(task_id)
        print(json.dumps(config, indent='\t'))

    def get_task_config(self, task_id):
        """
        config

        :param task_id: the task you want to get
        :returns: dict, task config
        """
        if not self.__is_task_registered(task_id):
            print(f'TaskManager: no task called \'{task_id}\'')
            sys.exit()
        task_path = self.__get_task_path(self.__get_task_brief(task_id))
        task_config = self.__get_task_config(
            os.path.join(task_path, f'{task_id}_config.json')
        )
        if not task_config['is_config']:
            print('WARNING: task not configured yet')
        return task_config

    def get_task_path(self, task_id):
        """
        get path according to task_id

        :param task_id:
        :returns: a set of task path
        """
        if not self.__is_task_registered(task_id):
            print(f'TaskManager: no task called \'{task_id}\'')
            sys.exit()
        task_path = self.__get_task_path(self.__get_task_brief(task_id))
        task_path_dict = {
            'task_path': task_path,
            'log_path': os.path.join(task_path, 'log.log'),
            'checkpoint_path': os.path.join(task_path, 'checkpoint'),
            'record_path': os.path.join(task_path, 'record'),
        }
        return task_path_dict

    def __get_task_brief(self, task_id):
        """
        get task brief from reg list
        """
        try:
            index = [item.get('task_id') for item in self.__register_list].index(task_id)
        except ValueError as err:
            print(f'TaskManager: task not found, {err}')
            sys.exit()
        return self.__register_list[index]

    def __get_task_path(self, task_brief):
        """
        get path from {'create_time', 'task_id'}
        """
        task_dirname = task_brief.get('create_time') + '_' + task_brief.get('task_id')
        task_path = os.path.join('task_record', task_dirname)
        return task_path


    def __write_task_config(self, config, config_path):
        """
        write config file
        """
        with open(config_path, 'w') as jsonfile:
            json.dump(config, jsonfile, indent='\t')

    def __read_task_config(self, config_path):
        """
        load config
        """
        with open(config_path, 'r') as jsonfile:
            try:
                config = json.load(jsonfile)
            except JSONDecodeError:
                print('TaskManager: Not valid json doc!')
                sys.exit()
        return config


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='task manager cmd')
    parser.add_argument("--new", "-n", action="store_true", help="create a new task")
    parser.add_argument("--list", "-l", action="store_true", help="show register list")
    parser.add_argument("--showconfig", "-s", action="store_true", help="show config of a task")
    parser.add_argument("--taskid", "-id", type=str, default='', help="an id of a task")
    args = parser.parse_args()

    tm = TaskManager()

    # new task
    if args.new:
        if args.taskid == '':
            print(f'TaskManager: task id required')
        else:
            tm.new_task(args.taskid)
            print(f'\'{args.taskid}\' created')

    # show task config
    elif args.showconfig:
        if args.taskid == '':
            print(f'TaskManager: taskid required')
        else:
            tm.display_task_config(args.taskid)

    # show register list
    elif args.list:
        tm.show_register_list()
            

# 改进其他任务通用性，比如文件夹结构配置