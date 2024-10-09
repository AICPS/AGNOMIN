import numpy as np
import pandas as pd
import wandb
print(wandb.__path__)

import sys
import os
from argparse import ArgumentParser
import random
import json
import difflib

import pickle
from tokenizer import tokenization
from collections import defaultdict


class Config():  
    def __init__(self, args):
        self.p = ArgumentParser(description='The parameters for ACFG pipeline.')
        self.p.add_argument('--src_name', type=str, default="", help="Name of the embeddings/labels files.")
        self.p.add_argument('--top_N', type=int, default=4096, help='Size of the ground truth label space.')
        self.p.add_argument('--exclude_arch', type=str, default="None", help="Name of the embeddings/labels files.")
        self.p.add_argument('--dataType', type=str, default="train-test", help="train-test or eval")
        self.p.add_argument('--pkl_path', type=str, default="bb_func_128_4096/bb_func_128_4096_label2id.pkl", help="label2id mapping for the model/only for evaluation purpose")
        args_parsed = self.p.parse_args(args)
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)
        
def read_data_file(datapath):
    X_data = None
    Y_data = []
    with open(datapath, 'r') as data_file:
        X_data = pd.read_table(data_file, engine='python', header=None, skiprows=1, sep= ':| ')
        drop_col = [n for n in range(1, X_data.shape[1], 2)]
        drop_col.append(0)
        X_data.drop(columns=drop_col, inplace=True)
    with open(datapath, 'r') as data_file:
        data_file.readline()
        for line in data_file:
            split_line = line.split(',')
            str_labels = split_line[:-1]
            str_labels.append(split_line[-1].split(' ')[0])
            Y_data.append([int(label) for label in str_labels])
            
    return X_data.values.astype(np.float32), Y_data

def perform_tokenization(src_name):

    function_names = pd.read_csv('embeddings/names_' + src_name + '.csv')
    name2kw = dict()
    with open('embeddings/names_tokenization.json', 'r') as name2kw_json:
        name2kw = json.load(name2kw_json)

    no_arch_func_names = []
    untokenized_names = set()

    for arch_func_name in function_names.to_string(header=False, index=False).split('\n'):
        splits = arch_func_name.lstrip(' ').split('.')
        no_arch_func_name = ''
        for string in splits[3:]:
            no_arch_func_name += string
        no_arch_func_names.append(no_arch_func_name)
        if not no_arch_func_name in name2kw.keys():
            untokenized_names.update(no_arch_func_name)
    
    if len(untokenized_names) > 0:
        new_tokenizations = tokenization.Function.processList(list(untokenized_names))
        for func in new_tokenizations:
            name2kw[func.name] = func.tokens_word
        json.dump(name2kw, 'embeddings/names_tokenization.json')
    
    Y_count_dict = defaultdict(int)
    Y_data = []
    for name in no_arch_func_names:
        Y_data.append(name2kw[name])
        for kw in name2kw[name]:
            Y_count_dict[kw] += 1
    return Y_data, Y_count_dict

#prunes all labels not found in the top_N. prunes all function names only containing tokens not found in the top_N.
def reduce_label_space(Y_data, Y_count_dict, top_N):
    # import pdb; pdb.set_trace()
    frequency = sorted(Y_count_dict.items(), key=lambda x:-x[1])
    top_n_tokens = [key for key, value in frequency[:top_N]]
    label2id = {label:id for id, label in enumerate(top_n_tokens)}
    for idx, label_list in enumerate(Y_data):
        Y_data[idx] = [label2id[label] for label in label_list if label in top_n_tokens]

    Y_data_final = []
    delete_idxs = []
    for idx, label_list in enumerate(Y_data):
        if label_list == []:
            delete_idxs.append(idx)
        else:
            Y_data_final.append(label_list)
    return Y_data_final, delete_idxs, label2id

def find_closest(words, query):
    if query in words:
        return 1
    else:
        closest_matches = difflib.get_close_matches(query, words, n=1, cutoff=0.0)
        if closest_matches:
            return closest_matches[0]
        else:
            return ""

def map_labels(Y_data, Y_count_dict, pickle_path):
    
    with open(pickle_path, 'rb') as label2id_file:
        label2id = pickle.load(label2id_file)
    Y_data_tokens = [token for func_name in Y_data for token in func_name]
    tokens2dict = {}
    for token in Y_data_tokens:
        if token in list(label2id.keys()):
            tokens2dict[token] = token
        else:
            tokens2dict[token] = find_closest(list(label2id.keys()), token)

    for idx, label_list in enumerate(Y_data):
        Y_data[idx] = [label2id[tokens2dict[label]] for label in label_list]
        
    return Y_data


if __name__ == '__main__':
    cfg = Config(sys.argv[1:])
    os.makedirs("dataFiles", exist_ok=True)
    if cfg.dataType == "eval":
        with open(f'dataFiles/{cfg.src_name}_{cfg.top_N}_tst.txt', 'w') as eval_data_file: 
            X_data = pd.read_csv('embeddings/embeddings_' + cfg.src_name + '.csv').to_numpy().astype(np.float32)
            names_ = pd.read_csv('embeddings/names_' + cfg.src_name + '.csv').to_numpy()

            Y_data, Y_count_dict = perform_tokenization(cfg.src_name)
            if cfg.top_N == -1:
                cfg.top_N = len(Y_count_dict)
            Y_data_final = map_labels(Y_data, Y_count_dict, cfg.pkl_path)
            assert len(X_data) == len(Y_data_final)
            
            eval_data_file.write(str(X_data.shape[0]) + ', ' + str(X_data.shape[1]) + ', ' + str(cfg.top_N) + '\n')
            for idx in range(X_data.shape[0]):
                if cfg.exclude_arch not in names_[idx][0]: 
                    eval_data_file.write(str(Y_data_final[idx]).strip('[],').replace(' ', '') + ' ')
                    for idx2 in range(X_data.shape[1]):
                        eval_data_file.write(str(idx2) + ':' + str(X_data[idx][idx2]) + ' ')
                    eval_data_file.write('\n')
  
    else:
        if not os.path.isfile(f'dataFiles/{cfg.src_name}_{cfg.top_N}_tst.txt') or not os.path.isfile(f'dataFiles/{cfg.src_name}_{cfg.top_N}_trn.txt'):           
            with open(f'dataFiles/{cfg.src_name}_{cfg.top_N}_trn.txt', 'w') as train_data_file: 
                X_data = pd.read_csv('embeddings/embeddings_' + cfg.src_name + '.csv').to_numpy().astype(np.float32)
                names_ = pd.read_csv('embeddings/names_' + cfg.src_name + '.csv').to_numpy()

                Y_data, Y_count_dict = perform_tokenization(cfg.src_name)
                if cfg.top_N == -1:
                    cfg.top_N = len(Y_count_dict)
                Y_data_final, delete_idxs, label2id = reduce_label_space(Y_data, Y_count_dict, cfg.top_N)

                #Save label2id for inference:
                with open(f'dataFiles/{cfg.src_name}_{cfg.top_N}_label2id.pkl', 'wb') as label2id_pickle_file: 
                    pickle.dump(label2id, label2id_pickle_file)

                X_data = np.delete(X_data, delete_idxs, 0)
                names_ = np.delete(names_, delete_idxs, 0)
                assert len(X_data) == len(Y_data_final)
                
                train_data_file.write(str(X_data.shape[0]*4//5) + ', ' + str(X_data.shape[1]) + ', ' + str(cfg.top_N) + '\n')
                for idx in range(X_data.shape[0]*4//5):
                    if cfg.exclude_arch not in names_[idx][0]: 
                        train_data_file.write(str(Y_data_final[idx]).strip('[],').replace(' ', '') + ' ')
                        for idx2 in range(X_data.shape[1]):
                            train_data_file.write(str(idx2) + ':' + str(X_data[idx][idx2]) + ' ')
                        train_data_file.write('\n')
                test_data_file = open(f'dataFiles/{cfg.src_name}_{cfg.top_N}_tst.txt', 'w')
                test_data_file_armel = open(f'dataFiles/{cfg.src_name}_{cfg.top_N}_armel_tst.txt', 'w') 
                test_data_file_amd64 = open(f'dataFiles/{cfg.src_name}_{cfg.top_N}_amd64_tst.txt', 'w') 
                test_data_file_i386 = open(f'dataFiles/{cfg.src_name}_{cfg.top_N}_i386_tst.txt', 'w') 
            
                armel_number = 0
                amd64_number = 0
                i386_number = 0

                print(f"sape_ {names_.shape} .  ")
                for idx in range(X_data.shape[0]*4//5, X_data.shape[0]):
                    if "armel" in names_[idx][0]:
                        armel_number = armel_number+1
                    if "amd64" in names_[idx][0]:
                        amd64_number = amd64_number+1
                    if "i386" in names_[idx][0]:
                        i386_number = i386_number+1
                print(f"total number of test cases is {X_data.shape[0] - (X_data.shape[0]*4//5)}")
                print(f"Test case # armel {armel_number} amd64 {amd64_number} i386 {i386_number}")
                test_data_file.write(str(X_data.shape[0]//5) + ', ' + str(X_data.shape[1]) + ', ' + str(cfg.top_N) + '\n')
                test_data_file_armel.write(str(armel_number) + ', ' + str(X_data.shape[1]) + ', ' + str(cfg.top_N) + '\n')
                test_data_file_i386.write(str(i386_number) + ', ' + str(X_data.shape[1]) + ', ' + str(cfg.top_N) + '\n')
                test_data_file_amd64.write(str(amd64_number) + ', ' + str(X_data.shape[1]) + ', ' + str(cfg.top_N) + '\n')
                for idx in range(X_data.shape[0]*4//5, X_data.shape[0]):
                    if "armel" in names_[idx][0]: 
                        test_data_file_armel.write(str(Y_data_final[idx]).strip('[],').replace(' ', '')  + ' ')
                        for idx2 in range(X_data.shape[1]):
                            test_data_file_armel.write(str(idx2) + ':' + str(X_data[idx][idx2]) + ' ')
                        test_data_file_armel.write('\n')
                    if "i386" in names_[idx][0]: 
                        test_data_file_i386.write(str(Y_data_final[idx]).strip('[],').replace(' ', '')  + ' ')
                        for idx2 in range(X_data.shape[1]):
                            test_data_file_i386.write(str(idx2) + ':' + str(X_data[idx][idx2]) + ' ')
                        test_data_file_i386.write('\n')
                    if "amd64" in names_[idx][0]: 
                        test_data_file_amd64.write(str(Y_data_final[idx]).strip('[],').replace(' ', '')  + ' ')
                        for idx2 in range(X_data.shape[1]):
                            test_data_file_amd64.write(str(idx2) + ':' + str(X_data[idx][idx2]) + ' ')
                        test_data_file_amd64.write('\n')
                    test_data_file.write(str(Y_data_final[idx]).strip('[],').replace(' ', '')  + ' ')
                    for idx2 in range(X_data.shape[1]):
                        test_data_file.write(str(idx2) + ':' + str(X_data[idx][idx2]) + ' ')
                    test_data_file.write('\n')
    print("done preprocessing")