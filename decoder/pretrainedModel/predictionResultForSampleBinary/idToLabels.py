import csv, sys
import pandas as pd
import os
import numpy as np
from argparse import ArgumentParser
import pickle


def load_label2id(filepath):
    with open(filepath, 'rb') as file:
        label2id = pickle.load(file)
    id2label = {v: k for k, v in label2id.items()}
    return id2label

def load_ids_from_csv(file_path):
    with open(file_path, newline='') as file:
        reader = csv.reader(file)
        next(reader) 
        return [[id.strip() for id in row if id.strip()] for row in reader]

def generate_labels(list_of_id_lists, id2label):
    return [[id2label.get(int(id), "") if id.isdigit() else "" for id in row] for row in list_of_id_lists]

def write_labels_to_csv(labels_lists, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Label'])
        writer.writerows(labels_lists)
        
def main():
    label2id_filepath = 'bb_func_128_4096_label2id.pkl'
    id2label = load_label2id(label2id_filepath)
    ids_file_path = 'renee-exp__predicion.csv' 
    id_list = load_ids_from_csv(ids_file_path)
    labels = generate_labels(id_list, id2label)
    output_csv_path = 'output_labels.csv'
    write_labels_to_csv(labels, output_csv_path)

if __name__ == '__main__':
    main()
