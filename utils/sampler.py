# -*- encoding: utf-8 -*-
'''
@File    :   sampler.py
@Time    :   2023/06/16 16:55:51
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
from typing import Iterator, List, Optional
import random
import math
import torch
from torch.utils.data import Sampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from transformers.tokenization_utils_base import BatchEncoding
import numpy as np
import copy


class BatchedTaskSampler(Sampler):

    def __init__(self, data_source, batch_size) -> None:
        
        super().__init__(data_source)

        self.data = data_source
        self.bs = batch_size

    
    def __iter__(self) -> Iterator:
        '''Assume we get (input_ids, labels, task). Task is encoded by labelencoder.'''
        task_data = self.data['task']
        task_data_index = list(range(len(task_data)))   # all index
        index_list = [] # save sampled index list
        task_index = {task: list(np.where(task_data==task)[0]) for task in np.unique(task_data)} # a dict to save each task index
        rest_task = list(np.unique(task_data))  # the list to save the task that has not been sampled

        while 1:
            
            if len(task_data_index) <=0:
                break
            
            random_task = random.sample(rest_task, 1)[0]    # randomly sample a task

            if len(task_index[random_task]) > self.bs:  # sample
                random_index = random.sample(task_index[random_task], self.bs)  # sample a batch of index
                index_list.append(random_index) # add sampled index to list
                for r_index in random_index:
                    task_data_index.remove(r_index)    # remove sampled index from all index list
                    task_index[random_task].remove(r_index)    # remove sampled index from corresponding task index list
            
            elif len(task_index[random_task]) > 0:  # directly add rest index
                random_index = copy.deepcopy(task_index[random_task])
                index_list.append(random_index) # add sampled index to list
                for r_index in random_index:
                    task_data_index.remove(r_index)    # remove sampled index from all index list
                    task_index[random_task].remove(r_index)
                rest_task.remove(random_task)

            else: # this task has been sampled, no rest index
                continue

        new_index, residual_index = [], []
        random.shuffle(index_list)  # make difference between epochs
        for meta_index_list in index_list:
            
            if len(meta_index_list) == self.bs:
                new_index += meta_index_list    # concat all index

            else:
                residual_index += meta_index_list   # save all incompleted batch
        
        for i in range(len(new_index)):
            new_index[i] = int(new_index[i])
        for i in range(len(residual_index)):
            residual_index[i] = int(residual_index[i])
        
        return iter(new_index + residual_index)


    def __len__(self):

        return len(self.data)



class DistributedBatchedTaskSampler(DistributedSampler):
    r"""
    Distributed Sampler that samples indices in a way that groups together features of the dataset of roughly the same
    length while keeping a bit of randomness.
    """

    # Copied and adapted from PyTorch DistributedSampler.
    def __init__(
        self,
        batch_size: int,
        dataset: Optional[Dataset] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.dataset = dataset
        self.bs = batch_size

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset['task']) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil((len(self.dataset['task']) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset['task']) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

    def __iter__(self) -> Iterator:
        # Deterministically shuffle based on epoch and seed
        task_data = self.dataset['task']
        task_data_index = list(range(len(task_data)))   # all index
        index_list = [] # save sampled index list
        task_index = {task: list(np.where(task_data==task)[0]) for task in np.unique(task_data)} # a dict to save each task index
        rest_task = list(np.unique(task_data))  # the list to save the task that has not been sampled

        while 1:
            
            if len(task_data_index) <=0:
                break
            
            random_task = random.sample(rest_task, 1)[0]    # randomly sample a task

            if len(task_index[random_task]) > self.bs:  # sample
                random_index = random.sample(task_index[random_task], self.bs)  # sample a batch of index
                index_list.append(random_index) # add sampled index to list
                for r_index in random_index:
                    task_data_index.remove(r_index)    # remove sampled index from all index list
                    task_index[random_task].remove(r_index)    # remove sampled index from corresponding task index list
            
            elif len(task_index[random_task]) > 0:  # directly add rest index
                random_index = copy.deepcopy(task_index[random_task])
                index_list.append(random_index) # add sampled index to list
                for r_index in random_index:
                    task_data_index.remove(r_index)    # remove sampled index from all index list
                    task_index[random_task].remove(r_index)
                rest_task.remove(random_task)

            else: # this task has been sampled, no rest index
                continue

        new_index, residual_index = [], []
        random.shuffle(index_list)  # make difference between epochs
        for meta_index_list in index_list:
            
            if len(meta_index_list) == self.bs:
                new_index += meta_index_list    # concat all index

            else:
                residual_index += meta_index_list   # save all incompleted batch
        
        for i in range(len(new_index)):
            new_index[i] = int(new_index[i])
        for i in range(len(residual_index)):
            residual_index[i] = int(residual_index[i])
        
        indices = new_index + residual_index

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[: (self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

