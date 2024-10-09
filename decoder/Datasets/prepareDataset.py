import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix
import numpy as np
import os, shutil, sys
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser

class Config():
    '''config for data processing pipeline.'''
    
    def __init__(self, args):
        self.p = ArgumentParser(description='The parameters for ACFG pipeline.')
        self.p.add_argument('--dataFilesDir', type=str, default="dataFiles/", help="Name of the directories where data files are stored.")
        self.p.add_argument('--destination_dir', type=str, default="./", help='Directory to save the dataset')
        self.p.add_argument('--allEntries', type=bool, default=False, help='Do you want to process all unprocessed data files in the dataFilesDir?')
        self.p.add_argument('--dataset_folders', type=str, default="bb_func_128_AGNOMIN_dataset_4096", help="This is used if we are not processing all entries in the datafile. we can put the dataset folder to be processed.")
        args_parsed = self.p.parse_args(args)
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

def create_memmap_from_file(file_name, memmap_embeddings_file, memmap_mask_file, dtype=np.int32):
    with open(file_name, 'r') as f:
        first_line = f.readline().strip()
        num_rows, num_cols = map(int, first_line.split())
        embeddings_memmap = np.memmap(memmap_embeddings_file, dtype=dtype, mode='w+', shape=(num_rows, num_cols))
        mask_memmap = np.memmap(memmap_mask_file, dtype=np.int8, mode='w+', shape=(num_rows, num_cols))
        embeddings_memmap[:] = 0
        mask_memmap[:] = 0

        for row_idx, line in enumerate(f):
            entries = line.strip().split()
            for entry in entries:
                col_idx, value = entry.split(':')
                try:
                    embeddings_memmap[row_idx, int(col_idx)] = float(value)
                    mask_memmap[row_idx, int(col_idx)] = 1  # Mark this position as having data
                except:
                    print(f"{file_name}_{memmap_embeddings_file}")
    embeddings_memmap.flush(), mask_memmap.flush()

    return (num_rows, num_cols)

def process_content(input_filename, dataType, dataset):
    with open(input_filename, 'r') as file:
        lines = file.read().strip().split("\n")    
    
    header = lines[0].split(", ")  
    first_file_header = f"{header[0]} {header[1]}"
    second_file_header = f"{header[0]} {header[2]}"

    first_file_content = [first_file_header]
    second_file_content = [second_file_header]

    for line in lines[1:]:  
        parts = line.split(" ")
        matrix_values = parts[0].split(",")
        vector_values = " ".join(parts[1:])

        matrix_line = " ".join([f"{val}:1" for val in matrix_values])
        second_file_content.append(matrix_line)
        first_file_content.append(vector_values)
    
    first_file_name = f"{dataset}/{dataType}_X_Xf.txt"
    second_file_name = f"{dataset}/{dataType}_X_Y.txt"

    with open(first_file_name, 'w') as file1:
        file1.write("\n".join(first_file_content))

    with open(second_file_name, 'w') as file2:
        file2.write("\n".join(second_file_content))


def adjust_values(input_filename):
    with open(input_filename, 'r') as file:
        lines = file.read().strip().split("\n")
    vector_lines = lines[1:]

    min_value = float('inf')
    # print(min_value)
    # Find the minimum value
    adjusted_lines = [lines[0]]  # Add the header back
    for line in vector_lines:
        adjusted_values = []
        values = line.split(" ")
        for value in values:
            if value !=  "":
                a, b = value.split(":")
                # min_value = max(1, float(b))
                min_value = 100*float(b)
                adjusted_b = str(min_value) 
                adjusted_values.append(f"{a}:{adjusted_b}")
        adjusted_line = " ".join(adjusted_values)
        adjusted_lines.append(adjusted_line)

    # Write the adjusted content back to a new file
    adjusted_file_name = input_filename
    with open(adjusted_file_name, 'w') as adjusted_file:
        adjusted_file.write("\n".join(adjusted_lines))

def cleanUp(folder):
    file_paths = [f"{folder}/trn_X_Xf.txt", f"{folder}/trn_X_Y.txt", f"{folder}/trn_embeddings.dat", f"{folder}/trn_mask_embeddings.dat", f"{folder}/tst_X_Xf.txt", f"{folder}/tst_X_Y.txt", f"{folder}/tst_embeddings.dat", f"{folder}/tst_mask_embeddings.dat"] 
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

def moveFiles(source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)

    files = os.listdir(source_dir)

    for file in files:
        new_filename = file
        subdirectory = new_filename.replace('_trn', '').replace('_tst', '').replace('.txt', '').replace('_label2id.pkl', '')
        print(subdirectory)
        subdirectory_path = os.path.join(destination_dir, subdirectory)
        
        if os.path.exists(os.path.join(subdirectory_path, new_filename)):
            print("already copied")
            continue
        os.makedirs(subdirectory_path, exist_ok=True)
        shutil.copy(os.path.join(source_dir, file), os.path.join(subdirectory_path, new_filename))
    print("Files have been successfully moved and renamed.")

def process_folder(folder):
    data_types=["trn", "tst"]
    if os.path.exists(f"{folder}/trn_filter_labels.txt"):
        print(f"already processed {folder}")
        return
    else:
        cleanUp(folder)
        for data_type in data_types:
            print(f"processing {folder} {folder}/{folder}_{data_type}.txt")
            if os.path.exists(f"{folder}/{folder}_{data_type}.txt"):
                print(f"Processing {data_type}")
                file_name = f"{folder}/{data_type}_X_Xf.txt"

                open(f"{folder}/trn_filter_labels.txt", "w")
                open(f"{folder}/tst_filter_labels.txt", "w")
                
                memmap_embeddings_file = f"{folder}/{data_type}_embeddings.dat"
                memmap_mask_file = f"{folder}/{data_type}_mask_embeddings.dat"

                process_content(f"{folder}/{folder}_{data_type}.txt", data_type, folder)

                print(f"Done processing {data_type}. Now creating memmaped file")
                adjust_values(file_name)
                
                if "tst" in data_type:
                    shape = create_memmap_from_file(file_name, memmap_embeddings_file, memmap_mask_file, dtype=np.float32)
                else:
                    shape = create_memmap_from_file(file_name, memmap_embeddings_file, memmap_mask_file, dtype=np.float32)
                print(f"memmapped file created and tested for {data_type}")



if __name__ == '__main__':
    """This is a data processing script. We assume users don't modify files names generated by the denDataFiles.py script.
        This script will process data files specified in under dataFiles directory and prepare train/test sets and all other necessary files
        and put them under the directory specified by the user. Names of datasets is usually follow 
        [pcode2vec embedding format]_[pcode2vec features]_[Label space size]_[optional - cpu architecture]
    """
    cfg = Config(sys.argv[1:])

    source_dir = cfg.dataFilesDir
    destination_dir = cfg.destination_dir

    moveFiles(source_dir, destination_dir)

    all_entries = os.listdir(destination_dir)

    excludeFolders = ['_dataProcessing', 'dataFiles', 'tokenizer', 'data_log', 'embeddings','paper_embeddings']
    # Filter out entries that are directories
    # folders = [entry for entry in all_entries if os.path.isdir(os.path.join(entry)) and entry not in excludeFolders]
    # folders = ["bb_func_128_final_512", "none_128_cfg2vec_armel_amd64_512_amd64", "none_128_cfg2vec_armel_amd64_512"]
    # folders = ["none_128_cfg2vec_i386_armel_amd64_512"]
    # folders = ["none_128_cfg2vec_armel_amd64_512_amd64", "none_128_cfg2vec_armel_amd64_512", "none_128_cfg2vec_armel_amd64_512_i386", "none_128_cfg2vec_armel_amd64_512_armel"]
    # import pdb; pdb.set_trace()
    if cfg.allEntries:
        folders = [entry for entry in all_entries if os.path.isdir(os.path.join(entry)) and entry not in excludeFolders]
    else:
        folders = cfg.dataset_folders.split(',')
   
    print(f" processing {folders}")
    

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_folder, folders)