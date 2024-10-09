import torch
import xclib.data.data_utils as data_utils
from dl_base import *
import numpy as np
import scipy.sparse as sp
from scipy.sparse import dok_matrix
import os
import argparse
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import AutoModel
from collections import OrderedDict
import wandb

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def getTrainDataDimention(file_name):
    with open(file_name, 'r') as file:
        # Read the first line to get the dimensions
        first_line = file.readline().strip()
        _, cols = map(int, first_line.split())
    return cols
      
def generate_sparse_matrix(file_name):
    with open(file_name, 'r') as file:
        # Read the first line to get the dimensions
        first_line = file.readline().strip()
        rows, cols = map(int, first_line.split())
        
        sparse_matrix = dok_matrix((rows, cols), dtype=np.float32)
        # import pdb; pdb.set_trace()

        row_index = 0
        for line in file:
            entries = line.strip().split()
            for entry in entries:
                col_index, value = entry.split(':')
                try:
                  sparse_matrix[row_index, int(col_index)-1] = float(value)
                except:
                  pass 
            row_index += 1
    sparse_matrix_csr = sparse_matrix.tocsr()
    
    return sparse_matrix_csr

def start(device, ngpus_per_node, args):
  # set seed
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  np.random.seed(args.seed)

  torch.cuda.set_device(device)
  nb_id = device
  # initialize the process group
  args.rank = my_rank = 0
  dataset = args.data_dir.split('/')[-1]
  results_dir = f'./Results/Bert_{args.num_layers}_{args.seed}/'
  expname = args.expname
  os.makedirs(results_dir, exist_ok=True)

  device = f'cuda:{nb_id}'
  torch.cuda.set_device(device)


  # Load Data
  # import pdb; pdb.set_trace()
  DATA_DIR = args.data_dir
  if not args.infer:
    trn_X_Y = generate_sparse_matrix(f'{DATA_DIR}/trn_X_Y.txt')
  tst_X_Y = generate_sparse_matrix(f'{DATA_DIR}/tst_X_Y.txt')
  args.maxlen=getTrainDataDimention(f'{DATA_DIR}/tst_X_Xf.txt')
  
  tst_shape_0, tst_shape_1 = tst_X_Y.shape[0], tst_X_Y.shape[1]
  if my_rank > 0: # only rank 0 does eval 
      tst_X_Y = None
  # import pdb; pdb.set_trace()

  if args.infer:
    A = 0.55; B = 1.5
    inv_prop = xc_metrics.compute_inv_propesity(tst_X_Y, A, B)
  else:
    A = 0.55; B = 1.5
    inv_prop = xc_metrics.compute_inv_propesity(trn_X_Y, A, B) 

  temp = np.fromfile('%s/tst_filter_labels.txt'%(DATA_DIR), sep=' ').astype(int)
  temp = temp.reshape(-1, 2).T
  tst_filter_mat = sp.coo_matrix((np.ones(temp.shape[1]), (temp[0], temp[1])), (tst_shape_0,tst_shape_1)).tocsr()

  if args.infer:
    padded_numy = tst_X_Y.shape[1]
    if (padded_numy % args.world_size != 0):
      padded_numy = args.world_size*((tst_X_Y.shape[1] + args.world_size -1)//args.world_size)
      if my_rank == 0:
        print("Rounding numy to",padded_numy)
  else:
    padded_numy = trn_X_Y.shape[1]
    if (padded_numy % args.world_size != 0):
      padded_numy = args.world_size*((trn_X_Y.shape[1] + args.world_size -1)//args.world_size)
      if my_rank == 0:
        print("Rounding numy to",padded_numy)

  numy_per_gpu = padded_numy // args.world_size
  if (numy_per_gpu % 16 != 0):
        padded_numy = args.world_size*16*((numy_per_gpu + 15)//16)
        numy_per_gpu = padded_numy // args.world_size
        if my_rank == 0:
          print("Rounding numy further to",padded_numy)
  if my_rank == 0:
    print("Final numy_per_gpu: ",numy_per_gpu)

  #Dataloaders
  if not args.infer:
    num_points = trn_X_Y.shape[0]
  else:
    num_points = tst_X_Y.shape[0] 

  start_label = my_rank*numy_per_gpu
  end_label = (my_rank+1)*numy_per_gpu
  # restrict the train dataset to only the labels that this rank is responsible for
  
  if not args.infer:
    trn_X_Y_rank = trn_X_Y.tocsc()[:,start_label:end_label].tocsr()
    trn_dataset = PreTokBertDataset(f'{DATA_DIR}', trn_X_Y_rank, num_points, args.maxlen, doc_type='trn')
  tst_dataset = PreTokBertDataset(f'{DATA_DIR}', tst_X_Y, tst_shape_0, args.maxlen, doc_type='tst')

 
  gbsz = args.batch_size*args.world_size 
  num_workers = 1
  if not args.infer:
    trn_loader = torch.utils.data.DataLoader( 
      trn_dataset,
      sampler=None,
      batch_size=gbsz,
      num_workers=num_workers,
      collate_fn=XCCollator(padded_numy, trn_dataset, my_rank, args.world_size, args.accum, gbsz//args.accum, True, args.default_impl),
      worker_init_fn=seed_worker, # need workers to use same seed so that batches are coordinated
      persistent_workers = False if (num_workers < 1) else True,
      shuffle=True,
      pin_memory=False)

  tst_loader = torch.utils.data.DataLoader( 
    tst_dataset, 
    batch_size=gbsz,
    num_workers=num_workers,
    collate_fn=XCCollator(padded_numy, tst_dataset, my_rank, args.world_size, args.accum, gbsz//args.accum, False, args.default_impl),
    worker_init_fn=seed_worker, # need workers to use same seed so that batches are coordinated
    persistent_workers = False if (num_workers < 1) else True,
    shuffle=False,
    pin_memory=False)

  if args.embedder == "doubleLinear":
    modules = [FP32Linear(args.maxlen, 512, bias=True), nn.Dropout(args.dropout), FP32Linear(512, 1024, bias=False) ] 
    args.bottleneck_dims = 1024
  elif args.embedder == "bert":
    modules = [BertEmbedder(embed_size=args.maxlen, hidden_size=1024, num_layers=args.num_layers), nn.Dropout(args.dropout), FP32Linear(1024, 1024, bias=False) ]
    args.bottleneck_dims = 1024
  else:
    modules = [FP32Linear(args.maxlen, 1024, bias=True), nn.Dropout(args.dropout)]
    args.bottleneck_dims = 1024

  if not args.infer:
    model = GenericModel(my_rank, args, trn_X_Y.shape[1], numy_per_gpu, args.batch_size, modules, device=device, name=expname, out_dir=f'{results_dir}/{args.data_dir}/{expname}')
  else:
    model = GenericModel(my_rank, args, tst_X_Y.shape[1], numy_per_gpu, args.batch_size, modules, device=device, name=expname, out_dir=f'{results_dir}/{args.data_dir}/{expname}')

  if args.compile:
      model.embed[0] = torch.compile(model.embed[0])
  model = model.cuda()
  #model.load()

  if args.world_size>1:
      model.embed = DDP(model.embed, device_ids=[my_rank%ngpus_per_node])

  if args.default_impl:
      trn_loss = BCELossMultiNodeDefault(model)
  else:
      trn_loss = BCELossMultiNode(model)


  model.epochs = args.epochs
  predictor = FullPredictor()
  evaluator = PrecEvaluator(model, args, tst_loader, predictor, filter_mat=tst_filter_mat, inv_prop=inv_prop)
  if args.infer:
      model.load(out_dir=f"pretrainedModel/Bert_5_9113/Datasets/bb_func_128_4096/renee-exp/")
      evaluator(trn_loss,out_dir="ResultsPaper", name=model.name)
  else:
      model.fit(trn_loader, trn_loss, 
          xfc_optimizer_class = apex.optimizers.FusedSGD,
          xfc_optimizer_params= {'lr': args.lr1, 'momentum': args.mo, 'weight_decay': args.wd1,  'set_grad_none': True},  
          tf_optimizer_class = apex.optimizers.FusedAdam,
          tf_optimizer_params= {'lr': args.lr2, 'eps': 1e-06, 'set_grad_none': True, 'bias_correction': True, 'weight_decay': args.wd2},
          epochs = args.epochs, warmup_steps = args.warmup,
          evaluator=evaluator,
          evaluation_epochs=5,
#           max_grad_norm=5.0)
          )

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='BERT XFC')
    parser.add_argument('--data-dir', type=str, default='',
                        help='Path to dataset directory')
    parser.add_argument('--custom-cuda', action='store_true', default=False,
                        help='Use custom_cuda kernels for fp16 training. Please ensure the custom kernels are optimized for the matmul sizes!!!')
    parser.add_argument('--default-impl', action='store_true', default=False,
                        help='Use default implemenation -- to get a sense of how much perf optimization gains over the default approach')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input per GPU batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--expname', type=str, default='renee-exp', metavar='N',
                        help='Name of exp')
    parser.add_argument('--device', type=int, default=0, metavar='N',
                        help='Single device training (default: None)')
    parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                    help='url used to set up distributed training; replace localhost with IP of rank 0 for multinode training')
    parser.add_argument('--rank', type=int, default=0, metavar='N',
                        help='Rank of node used in training (default: 0, should range from 0 to world-size -1)')
    parser.add_argument('--world-size', type=int, default=1, metavar='N',
                        help='Number of nodes used in training (default: 1)')
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                        help='seed (default: 42)')
    parser.add_argument('--bottleneck-dims', type=int, default=0, metavar='N',
                        help='bottleneck before fc layer (default: 0 means no bottleneck)')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='N',
                        help='Dropout prob (default: 0.5)')
    parser.add_argument('--lr1', type=float, default=0.0025, metavar='LR',
                        help='learning rate for xfc layer (default: 0.025)')
    parser.add_argument('--lr2', type=float, default=1e-5, metavar='LR',
                        help='learning rate for encoder (default: 1e-4)')
    parser.add_argument('--mo', type=float, default=0.9, metavar='MO',
                        help='momentum for xfc layer (default: 0.9)')
    parser.add_argument('--wd1', type=float, default=0.01, metavar='WD',
                        help='weight decay for xfc layer (default: 0.01)')
    parser.add_argument('--wd2', type=float, default=0.01, metavar='WD',
                        help='weight decay for encoder (default: 0.01)')
    parser.add_argument('--fp32encoder', action='store_true', default=False,
                        help='enable fp32 for encoder (default fp16)')
    parser.add_argument('--fp16xfc', action='store_true', default=False,
                        help='enable fp16 for xfc layer (default fp32)')
    parser.add_argument('--embedder', type=str, default="",
                        help='enable bert based embedder bert/doubleLinear/-')
    parser.add_argument('--noloss', action='store_true', default=False,
                        help='Skip loss computation (only accuracy is valid), saving memory/compute at extreme scale')
    parser.add_argument('--checkpoint-resume', type=str, default='checkpoint',
                        help='DIR to checkpoint each epoch/resume from most recent checkpoint')
    parser.add_argument('--infer', action='store_true', default=False,
                        help='Perform inference of pre-trained model')
    parser.add_argument('--warmup', type=int, default=140000, metavar='N',
                        help='number of steps for warmup (default: 140000)')
    parser.add_argument('--num_layers', type=int, default=1, metavar='N',
                        help='number of transformerEncoder layers (default 1)')
    parser.add_argument('--accum', type=int, default=1, metavar='N',
                        help='gradient accumulation steps in XFC layer to save memory (default: 1)')
    parser.add_argument('--pre-tok', action='store_true', default=False,
                        help='Use pre tokenized .dat files for dataset')
    parser.add_argument('--maxlen', type=int, default=32, metavar='N',
                        help='max seq length for transformer')
    parser.add_argument('--tf', type=str, default='distilbert-base-uncased', metavar='N',
                        help='encoder transformer type')
    parser.add_argument('--use-ngame-encoder', type=str, default='',
                        help='Path to NGAME pretrained encoder as initialization point for trainings')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Compile model using PyTorch 2.0')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb')
    parser.add_argument('--wandb')
    parser.add_argument('--wandb_project', type=str, default="decoder-model", help='wandb project')
    args = parser.parse_args()

    print(args)

    if args.use_wandb: 
      args.wandb = wandb.init(project=args.wandb_project , entity="aicpslab", config={}, name="Renee_Float_Time_%s_layers_%d_%s_%d_%d_%d_%d_%f"% (args.embedder, args.num_layers, args.data_dir, args.batch_size, args.epochs, args.maxlen, args.seed, args.dropout))         

    if args.device is not None:
        print("Specific device chosen. Starting single device training")
        start(args.device, 1, args)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.world_size *= ngpus_per_node
        if args.batch_size*args.world_size % args.accum != 0:
            print("Code currently expects  args.accum cleanly divides args.batch_size*args.world_size")
            exit(-1)
        print("Starting multi device training on ", args.world_size, " devices")
        mp.spawn(start, nprocs=ngpus_per_node, args=(ngpus_per_node, args), join=True)

if __name__ == '__main__':
     main()
