AGNOMIN: Encoder-decoder methodology for Function Name Reconstruction and Patch Situation
=====================
## To Get Started
In this repository, we offer a guide to run the encoder end of the AGNOMIN function name prediction. AGNOMIN's encoder and decoder are designed to be trained separately to ensure that each end of the pipeline can achieve maximum performance on its subtask, and to minimize the possibility of gradient propagation errors.

<a name="Running_AGNOMIN's_Encoder"></a>
### Installation Guide: Running AGNOMIN's_Encoder.
It is recommended to use an Anaconda virtual environment and Python 3.9. Here we provide an installation script on a Linux system with CUDA 11.8. The guide for installing Anaconda in Linux is [here](https://docs.anaconda.com/anaconda/install/linux/) also.

#### 1. Create Anaconda Working Environment
```sh
$ conda create --name agnomin python=3.9
$ conda activate agnomin
```
#### 2. Install Necessary Packages
Once the environment is created, then install the following modules with the following commands:
```sh
$ cd [your own project folder]
$ export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
$ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cudatoolkit=11.8 -c pytorch
$ python -m pip install torch-geometric==1.7.1
$ python -m pip install -r requirements_cfg2vec.txt
```

#### 3. Training the AGNOMIN Encoder

Once the environment setup is complete, you are ready to begin running the Python scripts contained in the scripts/ directory.
Running exp_agnomin.py trains an AGNOMIN model based on the specified command line arguments. You may use the '-h' or '--help' options with the script to learn more about each individual argument. 
Running exp_agnomin.py using a nonexistent '--pickle_path' argument will generate a .pkl file using the dataset specified by the '--dataset_path' argument at the specified pickle path. Depending on the size of the dataset, this may take a long time. After generating a .pkl for a dataset for the first time, it is recommended to always run the encoder training using the corresponding '--pickle_path'.
It is recommended to leave '--num_features_bb' and '--num_features_func' at 128 for best performance.

Sample usage:
```sh
$ python scripts/exp_agnomin.py --use_wandb \
 --epochs 100 \
 --learning_rate 0.0001 \
 --test_step 5 \
 --dataset_path AGNOMIN_dataset \
 --seed 1 \
 --pcode2vec bb_func \ 
 --pickle_path AGNOMIN_dataset.pkl \
 --num_features_bb 128  --num_features_func 128 \ 
 --pml agnomin_bb_func_128 \
 --test_size 0.2
```

Once either of these scripts has run, it will save a folder containing the model's weights (.pkth) and the command used to generate the model (.txt) at the directory specified by the --pml argument. 

Also included for your convenience is the pkl_gen.py script, which generates a .pkl file from a specified data folder without training a model or performing inference. This may be useful if you would like to generate multiple .pkl files before conducting any experiments.
Please make sure the '--num_features_bb' and '--num_features_func' are the same as the models you plan to train using the generated .pkl file. 

Sample usage:
```sh
$ python scripts/pkl_gen.py 
 --name '' \ 
 --dataset_path AGNOMIN_dataset \
 --pcode2vec bb_func \ 
 --pickle_path AGNOMIN_dataset.pkl \
 --num_features_bb 128 --num_features_func 128 \
```

### 4. Performing Inference Using a Trained Model

To perform inference using a trained AGNOMIN model, run the app_agnomin.py script.
The expected use case of the encoder is to generate embeddings that can be fed into AGNOMIN's decoder, but app_agnomin is also capable of performing function matching and similarity-based function name prediction. Note that this is not the recommended way to infer function names with AGNOMIN, and is provided solely for experimental purposes.

As with exp_agnomin.py, you may use the '-h' or '--help' options to learn more about each individual argument. 
app_agnomin.py's behavior is controlled by the value of the '--mode' argument, which determines what type of inference it performs.

'inference' creates an embedding for all functions contained within a single binary.
The path to the folder containing the binary should be specified by the '--p' argument.
This mode outputs three .csv files: embeddings_[inf_name].csv, labels_[inf_name].csv, and names_[inf_name].csv.
You may also control the directory the .csv files are written to using the '--inf_path' argument.
Caution with naming is advised, as the script will overwrite existing .csv files with the name inf_name/inf_path combination.

Sample Usage:
```sh
$ python scripts/app_agnomin.py \
 --mode inference \
 --p AGNOMIN_dataset/acct___ac-amd64.bin \
 --pcode2vec bb_func \
 --num_features_bb 128 --num_features_func 128 \
 --pml agnomin_bb_func_128 \
 --inf_path AGNOMIN_csvs \
 --inf_name agnomin_inference 
```

'inference_batch' creates an embedding for all functions contained within all binaries in a given dataset.
This dataset is specified by the '--pickle_path' argument, then by the '--dataset_path' argument if the .pkl file specified by the '--pickle_path' argument does not exist, in which case a new .pkl file will be created at that path using the dataset.
Similarly to 'inference', 'inference_batch' outputs three .csv files containing embeddings, labels, and names for all functions contained within all binaries for the given dataset.

Sample Usage:
```sh
$ python scripts/app_agnomin.py \
 --mode inference_batch \
 --pickle_path AGNOMIN_dataset.pkl \
 --pcode2vec bb_func \
 --num_features_bb 128 --num_features_func 128 \
 --pml agnomin_bb_func_128 \
 --inf_path AGNOMIN_csvs \
 --inf_name agnomin_inference
 ``` 

'func_match' attempts to determine which functions in two binaries are most closely matching by comparing the similarity of their embeddings. The directories of the two binaries should be specified using the '--p1' and '--p2' options. 
The results are printed to the file specified by the '--o' option.

Sample Usage:
```sh
$ python app_agnomin.py --mode func_match \
 --p1 ./AGNOMIN_dataset/acct___ac-amd64.bin \
 --p2 ./AGNOMIN_dataset/acct___ac-armel.bin \
 --pml ./saved_models/agnomin_bb_func_features_128 \
 --topk 10 --o result_fm.log 
```                                                    

'func_pred' attempts to determine which functions in a binary is most similar to the functions of every binary within a dataset. The directories of the binary should be specified using the '--p' option, while the directory of the dataset should be specified by the '--pickle_path' or '--dataset_path' options. The results are printed to the file specified by the '--o' option.

Sample Usage:
```sh
$ python app_agnomin.py --mode func_pred \
 --p ./AGNOMIN_dataset/acct___ac-amd64.bin \
 --pickle_path ./AGNOMIN_dataset.pkl \
 --pml ./saved_models/agnomin_bb_func_features_128 \
 --topk 10 --o result_fpd.log
```                                 
