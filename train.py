'''
Date: 2021-01-13 16:46:29
LastEditTime: 2021-01-13 20:26:18
Description: train model
FilePath: /DMGAN/train.py
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import os
import torch
import pandas as pd
from lib.utils import load_graph_data
from lib.seeds import seed
from model.dmgan_supervisor import DMGAN_Supervisor


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename) 
        supervisor_config['cuda_idx'] = args.cuda_idx
        supervisor = DMGAN_Supervisor(adj_mx=adj_mx, **supervisor_config)
        supervisor.train()


def set_cuda(args):
    DEVICE_ID = int(args.cuda_idx)
    torch.cuda.set_device(DEVICE_ID)
    gpu_list = str(DEVICE_ID)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    device = torch.device("cuda:{}".format(DEVICE_ID) if torch.cuda.is_available() else "cpu")
    print('device: ', device)



if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/DMGAN_pems03.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--cuda_idx', default=0, type=int, help='Set device idx')
    args = parser.parse_args()
    """set random seed"""
    # seed(2022)
    seed(1997)
    """set cuda idx"""
    set_cuda(args)    
    
    main(args)
