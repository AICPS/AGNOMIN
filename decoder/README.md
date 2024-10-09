# AGNOMIN Decoder
## Requirements

Run the below command, this will create a new conda environment with all the dependencies required to run Renee.

```bash
bash install1.sh
conda activate AGNOMIN_DECODER
bash install2.sh
```

## Data Preparation
Full Dataset will be released during the Artifact Evaluation. It is too big to attack it here. We have, however included sample datasets in the encoder folder, and the their function embedding (the result from the AGNOMIN Encoder under the `Datasets` directoty. )


To do a train test split, simply run `getDataFiles.py` script under the `Datasets` directory.:
```bash
cd Datasets
python genDataFiles.py --src_name [name of embedding file generated for the dataset. Assumin it is under embeddings/ directory] --top_N [number of label spaces.] --dataType [specify this to 'eval' if we are preping the dataset for evaluation purpose only, otherwise leave it empty] \
python genDataFiles.py --src_name bb_func_128_AGNOMIN_dataset --top_N 4096 --dataType eval # for eval only
# or
python genDataFiles.py --src_name bb_func_128_AGNOMIN_dataset --top_N 4096 # for train/test dataset 
```

To preprocess the dataset and prepare it for the model, please use `prepareDataset.py` script under the `Datasets` directory.:
```bash
python prepareDataset.py  --dataset_folders bb_func_128_AGNOMIN_dataset_4096
```

Above command will create a folder named `Datasets/bb_func_128_AGNOMIN_dataset_4096`, now we can refer to this dataset directory in our training script to train our decoder.

## Training

To train our decoder, we can use the followoing command. (make sure you modify `data-dir` argument accordingly)
```bash
python main.py
  --epochs 400 
  --num_layers 5 
  --embedder bert 
  --batch-size 64 
  --fp32encoder 
  --lr1 0.001 
  --lr2 5e-5
  --warmup 1000 
  --data-dir Datasets/bb_func_128_AGNOMIN_dataset_4096 
  --dropout 0.1 
  --wd1 0.001 
  --wd2 0.001 
  --seed 689 
  --compile 
```
To change hyperparameters, you can refer to the various arguments provided in `main.py` file or you can do `python main.py --help` to list out the all the arguments.

In this submission, we haven't provided the training dataset as it is huge in size. However we have a pretrained model under `pretrainedModel` directory. 

## Evaluation
To evaluate our decoder, we can use the followoing command. (make sure you modify `data-dir` argument accordingly)
```bash
python main.py 
  --infer 
  --num_layers 5 
  --embedder bert 
  --batch-size 64 
  --fp32encoder 
  --lr1 0.001 
  --lr2 5e-5
  --warmup 1000 
  --data-dir Datasets/bb_func_128_AGNOMIN_dataset_4096 
  --dropout 0.1 
  --wd1 0.001 
  --wd2 0.001 
  --seed 689 
  --compile 
```
The prediction result and predicted label ids can be found under `ResultsPaper` directory.
The above command only provides us with the predicted label ids and scores. However, we have provided the neccessary scripts and the predicted labels under  `pretrainedModel/predictionResultForSampleBinary` directory for the sample dataset. 
You can also use `pretrainedModel/predictionResultForSampleBinary/idToLabels.py` script to generate the labels from the ids.

Note that, we need to have `bb_func_128_4096_label2id.pkl` file to generate the files. This file is generated when we prepare the dataset. This file id dependent on the training dataset and the label space. 

## References
Our decoder is inspired by a multi-lable classifier known us Renee. We have explained in the paper how our model made significant advancements. The original code for renee can be found here: `https://github.com/microsoft/renee`