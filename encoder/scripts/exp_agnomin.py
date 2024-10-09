import os
import random
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from core.models import AGNOMINEncoder, cfg2vecGoG, GeminiACFG
import pickle as pkl
import random, torch
from argparse import ArgumentParser
from collections import Counter
from math import sqrt
from pathlib import Path

import pandas as pd
from core.acfg_parser import ACFGDataset
from core.trainer import HSNTrainer, MultiArchSampler
from kmeans_pytorch import kmeans
from torch_geometric.data import DataLoader

import numpy as np
from time import perf_counter, gmtime, strftime


class Config():
    '''config for ACFG pipeline.'''
    
    def __init__(self, args):
        self.p = ArgumentParser(description='The parameters for ACFG pipeline.')
        self.p.add_argument('--dataset_path', type=str, default="AGNOMIN_dataset", help="Path to dataset source folder.")
        self.p.add_argument('--pickle_path', type=str, default="AGNOMIN_dataset.pkl", help="Path to the dataset pickle file.")
        self.p.add_argument('--seed', type=int, default=random.randint(0,2**32), help='Random seed.')

        self.tg = self.p.add_argument_group('Training Config')
        self.tg.add_argument('--learning_rate', default=0.001, type=float, help='The initial learning rate for GCN/GMN.')
        self.tg.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
        self.tg.add_argument('--batch_size', type=int, default=32, help='Number of graphs in a batch.')
        self.tg.add_argument('--device', type=str, default="cuda", help='The device to run on models, cuda is default.')
        self.tg.add_argument('--model', type=str, default="agnomin", help="Model to be used (agnomin, acfg, cfg2vecGoG) agnomin is default.")
        self.tg.add_argument('--num_layers', type=int, default=3, help='Number of hidden layers to create for the model.')
        self.tg.add_argument('--num_features_bb', type=int, default=128, help="The initial dimension of each acfg node.")
        self.tg.add_argument('--num_features_func', type=int, default=512, help="(update)")
        self.tg.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    
        self.tg.add_argument('--pml', type=str, default="./saved_models/")
        self.tg.add_argument('--included_archs', type=str, default='i386,amd64,armel')
        self.tg.add_argument('--test_step', type=int, default=10, help='Number of epochs before testing the model.')
        self.tg.add_argument('--layer_spec', type=str, default='32,32,32', help='String of dimensions for hidden layers.')
        self.tg.add_argument('--test_size', type=float, default=0.2, help='Test set size proportion if doing train-test split.')
        self.tg.add_argument('--architectures', type=str, default='i386,amd64,armel', help='String of architectures for parsing.')
        self.tg.add_argument('--tolerance', type=int, default=0, help="Tolerance count for early stopping.")
        self.tg.add_argument('--topk', type=int, default=10, help="topk.")
        self.tg.add_argument('--num_clusters', type=int, default=3, help='Number of clusters for clustering functions.')
        self.tg.add_argument('--pcode2vec', type=str, default="none", help='[none|bb|func|bb_func]')
        self.tg.add_argument('--use_wandb', action='store_true', help='Use wandb')
        self.tg.add_argument('--wandb_project', type=str, default="AGNOMIN_wandb", help='wandb project')

        self.evaluate_group = self.p.add_argument_group('Evaluate Config')
        self.evaluate_group.add_argument('--eval_only', type=bool, default=False, help='Evaluate the model only (model must be loaded).')
        self.evaluate_group.add_argument('--metrics_path', type=str, default="./metrics/", help="Path to the metrics folder.")

        args_parsed = self.p.parse_args(args)
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)
        
        self.dataset_path = Path(self.dataset_path).resolve()
        self.pickle_path = Path(self.pickle_path).resolve()
        self.pml = Path(self.pml).resolve()
        self.metrics_path = Path(self.metrics_path).resolve()
        self.metrics_path.mkdir(exist_ok=True)
        

def read_dataset(cfg):
    if cfg.pickle_path.exists():
        dataset = pd.read_pickle(cfg.pickle_path)

    else:
        dataset = ACFGDataset(cfg.pcode2vec)
        dataset.load(cfg.dataset_path, cfg.num_features_bb, cfg.num_features_func)                
        with open(str(cfg.pickle_path), 'wb') as f:
            pkl.dump(dataset, f)
            print('Dumped pkl.')
    
    if cfg.included_archs == '':
        dataset.pack4agnomin()
    else:
        print(cfg.included_archs.split(','))
        dataset.pack4agnomin(cfg.included_archs.split(','))

    return dataset


if __name__ == "__main__":
    '''
        Usage Example:
            python exp_acfg_allstar.py --dataset_path /AGNOMIN_dataset
            --pickle_path final_dataset_db.pkl --seed 1 --device cuda --epochs 100 --batch_size 4 
            --use_wandb --pml "./saved_model/final_dataset" --wandb_project AGNOMIN_wandb final_dataset --architectures 'armel, amd64, i386, mipsel'
    '''
    cfg = Config(sys.argv[1:])
    print("reading dataset.")
    dataset = read_dataset(cfg)
    print("read dataset.")
    train_set, test_set, eval_set = dataset.split_dataset_by_package(cfg.test_size, cfg.seed)
    print(dataset.idx2kw)


    train_data_loader = DataLoader(train_set, batch_sampler=MultiArchSampler(train_set))
    valid_data_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    test_data_loader  = DataLoader(test_set,  batch_size=cfg.batch_size, shuffle=True)

    torch.cuda.memory._record_memory_history(True, device=torch.device('cuda'))
    if cfg.model == 'acfg':
        model = GeminiACFG(cfg.num_layers, cfg.layer_spec, cfg.num_features_bb, cfg.num_features_func, cfg.dropout, cfg.pcode2vec).to(cfg.device)
    elif cfg.model == 'cfg2vec':
        model = cfg2vecGoG(cfg.num_layers, cfg.layer_spec, cfg.num_features_bb, cfg.dropout, cfg.pcode2vec).to(cfg.device)
    else: 
        model = AGNOMINEncoder(cfg.num_layers, cfg.layer_spec, cfg.num_features_bb, cfg.num_features_func, cfg.dropout, cfg.pcode2vec).to(cfg.device)
    
    trainer = HSNTrainer(cfg, model, thunk_idx=dataset.func2idx['thunk']) # for siamese based network 
    # trainer.set_cluster_information(cluster_ids_x_test, cluster_ids_x_train, cluster_centers_test, cluster_centers_train)
    import pprint
    if cfg.eval_only:
        print("Loading already trained model.")
        trainer.load_model()     
    else:
        print("Training new Model.")
        
        start_train = perf_counter()
        trainer.train_model(train_data_loader, valid_data_loader, test_data_loader, idx2kw=dataset.idx2kw)
        end_train = perf_counter()
        print(f"Trained the model in " + strftime("%H:%M:%S", gmtime(end_train - start_train)))
        print(f"which equals {end_train - start_train:0.4f} seconds")

        trainer.save_model()

    with open('./paper_eval_log/eval_model_%s.log'%(str(cfg.pml).split("/")[-1]), "w") as log_file:
        test_data_loader  = DataLoader(test_set,  batch_size=1, shuffle=True)
        '''
        mipsel
        '''
        if "mipsel" in cfg.included_archs:
            log_file.write("mipsel-----\n")
            eval_results = {}
            mipsel_eval_set = [mipsel_bin for mipsel_bin in test_set if mipsel_bin.archi == "mipsel"]
            eval_data_loader  = DataLoader(mipsel_eval_set,  batch_size=1, shuffle=True)
            _, _, eval_hits, eval_loss, _ = trainer.inference(eval_data_loader, no_loss=True, eval_set=test_data_loader)
                
                # do evaluation.  
            for k in range(1, cfg.topk+1):
                eval_results['test_mipsel/p@%d'%(k)] = eval_hits[k-1]
            eval_results['test_mipsel/loss'] = eval_loss
            # do function name prediction candidate list.
            pprint.pprint(eval_results, log_file)
            log_file.write("mipsel-----\n\n")

        ''' 
            arm 
        '''
        if "armel" in cfg.included_archs:
            log_file.write("armel-----\n")    
            eval_results = {}
            arm_eval_set = [arm_bin for arm_bin in test_set if arm_bin.archi == "armel"]
            eval_data_loader  = DataLoader(arm_eval_set,  batch_size=1, shuffle=True)
            _, _, eval_hits, eval_loss, _ = trainer.inference(eval_data_loader, no_loss=True, eval_set=test_data_loader)
                
            # do evaluation.  
            for k in range(1, cfg.topk+1):
                eval_results['test_armel/p@%d'%(k)] = eval_hits[k-1]
            #eval_results['test_armel/loss'] = eval_loss
            # do function name prediction candidate list.
            pprint.pprint(eval_results, log_file)
            log_file.write("armel-----\n\n")
            
        ''' 
            i386 
        '''
        if "i386" in cfg.included_archs:
            log_file.write("i386-----\n") 
            eval_results = {}
            i386_eval_set = [i386_bin for i386_bin in test_set if i386_bin.archi == "i386"]
            eval_data_loader  = DataLoader(i386_eval_set,  batch_size=1, shuffle=True)
            _, _, eval_hits, eval_loss, _ = trainer.inference(eval_data_loader, no_loss=True, eval_set=test_data_loader)
            # do evaluation.        
            for k in range(1, cfg.topk+1):
                eval_results['test_i386/p@%d'%(k)] = eval_hits[k-1]
            #eval_results['test_i386/loss'] = eval_loss
            # do function name prediction candidate list.
            pprint.pprint(eval_results, log_file)
            log_file.write("i386-----\n\n")

        ''' 
            amd64 
        '''
        if "amd64" in cfg.included_archs:
            log_file.write("amd64-----\n") 
            eval_results = {}
            amd64_eval_set = [amd64_bin for amd64_bin in test_set if amd64_bin.archi == "amd64"]
            eval_data_loader  = DataLoader(amd64_eval_set,  batch_size=1, shuffle=True)

            _, _, eval_hits, eval_loss, _ = trainer.inference(eval_data_loader, no_loss=True, eval_set=test_data_loader)
            # do evaluation.        
            for k in range(1, cfg.topk+1):
                eval_results['test_amd64/p@%d'%(k)] = eval_hits[k-1]
            #eval_results['test_amd64/loss'] = eval_loss
            # do function name prediction candidate list.
            pprint.pprint(eval_results, log_file)
            log_file.write("amd64-----\n\n")

        log_file.write("total-----\n") 
        eval_results = {}
        eval_data_loader  = DataLoader(test_set,  batch_size=1, shuffle=True)
        _, _, eval_hits, eval_loss, _ = trainer.inference(eval_data_loader, no_loss=True)
        # do evaluation.        
        for k in range(1, cfg.topk+1):
            eval_results['test_total/p@%d'%(k)] = eval_hits[k-1]
        #eval_results['test_total/loss'] = eval_loss
        # do function name prediction candidate list.
        pprint.pprint(eval_results, log_file)
        log_file.write("total-----\n\n")
