import os
import random
import sys

sys.path.append(os.path.dirname(sys.path[0]))
import pickle as pkl
import random, torch
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from core.acfg_parser import ACFGDataset

import numpy as np
import os


class Config():
    '''config for ACFG pipeline.'''
    
    def __init__(self, args):
        self.p = ArgumentParser(description='The parameters for the .pkl generation script.')
        self.p.add_argument('--dataset_path', type=str, default="AGNOMIN_dataset", help="Path to dataset source folder.")
        self.p.add_argument('--pickle_path', type=str, default="AGNOMIN_dataset.pkl", help="Path to the dataset pickle file.")
        self.p.add_argument('--num_features_bb', type=int, default=128, help="The initial dimension of each acfg node.")
        self.p.add_argument('--num_features_func', type=int, default=128, help="The initial dimension of each acfg node.")
        self.p.add_argument('--pcode2vec', type=str, default="none", help='[none|bb|func|bb_func]')

        args_parsed = self.p.parse_args(args)
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)
        
        self.dataset_path = Path(self.dataset_path).resolve()
        self.pickle_path = Path(self.pickle_path).resolve()


def read_dataset(cfg):
    print(cfg.pickle_path)
    if cfg.pickle_path.exists():
        print('pkl file exists at this path.')

    else:
        dataset = ACFGDataset(cfg.pcode2vec)
        dataset.load(cfg.dataset_path, cfg.num_features_bb, cfg.num_features_func)

        with open(str(cfg.pickle_path), 'wb') as f:
            pkl.dump(dataset, f)
            print('pkl file dumped successfully.')

if __name__ == "__main__":
    cfg = Config(sys.argv[1:])
    read_dataset(cfg)
    


    
    
    

