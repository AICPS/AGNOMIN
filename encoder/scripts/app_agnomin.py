'''
    This material is based upon work supported by the
    Defense Advanced Research Projects Agency (DARPA)
    and Naval Information Warfare Center Pacific
    (NIWC Pacific) under Contract Number N66001-20-C-4024.

    The views, opinions, and/or findings expressed are
    those of the author(s) and should not be interpreted
    as representing the official views or policies of
    the Department of Defense or the U.S. Government.

    Distribution Statement "A" (Approved for Public Release,
    Distribution Unlimited) 
'''

import os
import random
import sys
import pprint
from pathlib import Path

sys.path.append(os.path.dirname(sys.path[0]))
from core.models import AGNOMINEncoder, GeminiACFG, cfg2vecGoG
import random
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import pickle as pkl
from core.acfg_parser import ACFGDataset
from core.trainer import HSNTrainer
from torch_geometric.data import DataLoader


class Config():
    '''config for ACFG pipeline.'''
    
    def __init__(self, args):
        self.p = ArgumentParser(description='The parameters for ACFG pipeline.')
        self.p.add_argument('--dataset_path', type=str, default="AGNOMIN_dataset", help="Path to dataset source folder.")
        
        self.p.add_argument('--mode', type=str, default="func_match", help="inference/func_match/func_pred for function name matching/function name prediction")
        self.p.add_argument('--model', type=str, default="agnomin", help='agnomin/acfg/cfg2vecGoG')
        self.p.add_argument('--p', type=str, default="", help="Path to FEHG features of binary.")
        self.p.add_argument('--p1', type=str, default="", help="Path to FEHG features of binary 1.")
        self.p.add_argument('--p2', type=str, default="", help="Path to FEHG features of binary 2.")
        self.p.add_argument('--pickle_path', type=str, default="./AGNOMIN_dataset.pkl", help="Path to the pickle folder.")
        self.p.add_argument('--pml', type=str, default="./saved_models/agnomin_bb_func_num_features_128", help="Path to the saved model.")
        
        self.p.add_argument('--batch_size', type=int, default=16, help='Number of graphs in a batch.')
        self.p.add_argument('--seed', type=int, default=random.randint(0,2**32), help='Random seed.')
        self.p.add_argument('--topk', type=int, default=10, help="k is used to decide how many candidate function names for each function name in the stripped binary.")
        
        self.p.add_argument('--pcode2vec', type=str, default="none", help='[none|bb|func|bb_func]')
        self.p.add_argument('--num_layers', type=int, default=3, help='Number of hidden layers to create for the model.')
        self.p.add_argument('--num_features_bb', type=int, default=128, help="The number of features for each basic block.")
        self.p.add_argument('--num_features_func', type=int, default=128, help="The number of features for each function.")
        self.p.add_argument('--dropout', type=float, default=0.1, help='Dropout')
        self.p.add_argument('--learning_rate', default=0.001, type=float, help='The initial learning rate for GCN/GMN.')
        self.p.add_argument('--layer_spec', type=str, default='32,32,32', help='String of dimensions for hidden layers.')
        self.p.add_argument('--device', type=str, default="cuda", help='The device to run on models, cuda is default.')
        self.p.add_argument('--metrics_path', type=str, default="./metrics/", help="Path to the metrics folder.")
        self.p.add_argument('--inf_path', type=str, default="./embeddings/", help="Path to the folder to place inferred embeddings.")
        self.p.add_argument('--inf_name', type=str, default="agnomin", help="Name of inferred embeddings.")
        self.p.add_argument('--o', type=str, default="", help="Path to the result log.")
        
        self.p.add_argument('--use_wandb', action='store_true', help='Use wandb')
        self.p.add_argument('--wandb_project', type=str, default="AGNOMIN_wandb", help='wandb project')



        args_parsed = self.p.parse_args(args)
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)
        
        self.dataset_path = Path(self.dataset_path).resolve()
        self.pml = Path(self.pml).resolve()
        self.metrics_path = Path(self.metrics_path).resolve()
        self.pickle_path = Path(self.pickle_path).resolve()
        self.p1 = Path(self.p1).resolve()
        self.p2 = Path(self.p2).resolve()
        self.p = Path(self.p).resolve()
        self.o = Path(self.o).resolve()


def get_dataset(path, num_features_bb, num_features_func):
    dataset = ACFGDataset(cfg.pcode2vec)
    dataset.load_1_binary(str(path), num_features_bb, num_features_func)
    dataset.pack4agnomin()
    return dataset 

def get_dataset_from_pickle(path):
    dataset = pd.read_pickle(str(path))
    dataset.pack4agnomin()
    return dataset 

def write_result(log_file, results):
    results_key = list(results.keys())
    for key in results_key:
        spec = key.split(".")
        log_file.write('{0:16}\n'.format("Package name  : {0}".format(spec[2])))
        log_file.write('{0:16}\n'.format("Binary file   : {0}".format(spec[3])))
        log_file.write('{0:16}\n'.format("Function name : {0}".format(spec[4])))
        log_file.write('{0:16}{1:19} {2:50}\n'.format("TopK matches  : ", "Similarity Score", "Function Name"))
        scores = list(results[key].keys())
        k = 0
        for score in scores:
            matched_names = results[key][score]
            for matched_name in matched_names:
                if k < cfg.topk:
                    log_file.write('{0:16}{1:>19} {2:50} \n'.format('', "{0}:".format(abs(score)), "{0}".format(matched_name)))
                    k+=1
                else:
                    break
            if k >= cfg.topk:
                break

if __name__ == "__main__":
    '''
        1. Usage:
            $ python app_agnomin.py --mode [func_match/func_pred] --p1 [path to 1st binary] --p2 [path to 2nd binary if mode == func_match] --k [number]

        2. Examples:
            $ python app_agnomin.py --mode inference --p1 ./AGNOMIN_dataset/acct___ac-amd64.bin
                                                      --pml ./saved_models/agnomin_bb_func_features_128
                                                      --topk 10 --o result_inf
            $ python app_agnomin.py --mode func_match --p1 ./AGNOMIN_dataset/acct___ac-amd64.bin
                                                      --p2 ./AGNOMIN_dataset/acct___ac-armel.bin
                                                      --pml ./saved_models/agnomin_bb_func_features_128
                                                      --topk 10 --o result_fm.log
            $ python app_agnomin.py --mode func_pred --p ./AGNOMIN_dataset/acct___ac-amd64.bin
                                                     --pickle_path ./AGNOMIN_dataset.pkl
                                                     --pml ./saved_models/agnomin_bb_func_features_128
                                                     --topk 10 --o result_fpd.log
    '''

    cfg = Config(sys.argv[1:])
    
    if cfg.model == "acfg":
        model = GeminiACFG(cfg.num_layers, cfg.layer_spec, cfg.num_features_bb, cfg.num_features_func, cfg.dropout, cfg.pcode2vec).to(cfg.device)
    elif cfg.model == "cfg2vecGoG":
        model = cfg2vecGoG(cfg.num_layers, cfg.layer_spec, cfg.num_features_bb, cfg.dropout, cfg.pcode2vec).to(cfg.device)
    else:
        model = AGNOMINEncoder(cfg.num_layers, cfg.layer_spec, cfg.num_features_bb, cfg.num_features_func, cfg.dropout, cfg.pcode2vec).to(cfg.device)

    if cfg.mode == "inference":
        if not cfg.p.exists():
            raise ValueError("The scripts expects one folder") 

        dataloader = DataLoader(get_dataset(cfg.p, cfg.num_features_bb, cfg.num_features_func).end2end_dataset, batch_size=cfg.batch_size)
        trainer = HSNTrainer(cfg, model)
        trainer.load_model()
        trainer.export_embeddings_labels_names_csv(cfg.inf_name, dataloader, cfg.inf_path)

    elif cfg.mode == "inference_batch":
        dataloader = None 
        if not cfg.pickle_path.exists():
            if not cfg.dataset_path.exists():
                raise ValueError("Neither a valid pickle_path nor a valid dataset_path were provided.")
            dataset = ACFGDataset(cfg.pcode2vec)
            dataset.load(cfg.dataset_path, cfg.num_features_bb, cfg.num_features_func)

            with open(str(cfg.pickle_path), 'wb') as f:
                pkl.dump(dataset, f)

            dataset.pack4agnomin()
            dataloader = DataLoader(dataset.end2end_dataset, batch_size=cfg.batch_size)
            
        else:            
            dataloader = DataLoader(get_dataset_from_pickle(cfg.pickle_path).end2end_dataset, batch_size=cfg.batch_size)
        
        trainer = HSNTrainer(cfg, model)
        trainer.load_model()
        trainer.export_embeddings_labels_names_csv(cfg.inf_name, dataloader, cfg.inf_path)


    elif cfg.mode == "func_match":
        if not cfg.p1.exists() or not cfg.p2.exists():
            raise ValueError("The scripts expects two folders")

        dataloader1 = DataLoader(get_dataset(cfg.p1, cfg.num_features_bb, cfg.num_features_func).end2end_dataset, batch_size=cfg.batch_size)
        dataloader2 = DataLoader(get_dataset(cfg.p2, cfg.num_features_bb, cfg.num_features_func).end2end_dataset, batch_size=cfg.batch_size)
        
        trainer = HSNTrainer(cfg, model)
        trainer.load_model()
        results = trainer.do_func_matching(dataloader1, dataloader2, True)
        
        with open(str(cfg.o), "w") as log_file:
            write_result(log_file, results)
                
    elif cfg.mode == "func_pred":
        if not cfg.p.exists():
            raise ValueError("The scripts expects one folder")

        dataloader1 = DataLoader(get_dataset(cfg.p, cfg.num_features_bb, cfg.num_features_func).end2end_dataset, batch_size=cfg.batch_size)
        dataloader2 = DataLoader(get_dataset_from_pickle(cfg.pickle_path).end2end_dataset, batch_size=cfg.batch_size)

        trainer = HSNTrainer(cfg, model)
        trainer.load_model()
        results = trainer.do_func_matching(dataloader1, dataloader2, True)
        
        with open(str(cfg.o), "w") as log_file:
            write_result(log_file, results)
            

    else:
        raise ValueError("set mode to appropriate value: inference/func_match/func_pred for embedding inference/function name matching/function name prediction")