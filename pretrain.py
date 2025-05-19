"""
Train a diffusion model for recommendation
"""
from tqdm import tqdm
import argparse
from ast import parse
import os
import time
import numpy as np
import copy
import re
from models import *
from utils import *
from utils import const, utils


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import scipy.sparse as sp

from models.Transformer import MultiLayerCrossAttention as former
import models.gaussian_diffusion as gd
# from models.Autoencoder import AutoEncoder as AE
# from models.Autoencoder import compute_loss
# from models.DNN import DNN
# import evaluate_utils
# import data_utils
from copy import deepcopy

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) #gpu
np.random.seed(random_seed) #numpy
random.seed(random_seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn


def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def parse_global_args(parser: argparse.ArgumentParser):
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--device', type=str, default='cuda:0,1,2,3')
    parser.add_argument('--random_seed', type=int, default=20230601)
    parser.add_argument('--time', type=str, default='none')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--test_path', type=str, default="")

    parser.add_argument('--data', type=str, default='KuaiSAR')
    parser.add_argument('--model', type=str, default='UniSAR')
    # parser.add_argument('--eval_batchsize', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
    
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adagrad', help='choice of optimizer')#
    # parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
    parser.add_argument('--save_path', type=str, default='/home/liuxiangxi/WJY/zhaorongchen/baseline/OurModel/diff_checkpoint', help='save model path')
    parser.add_argument('--log_name', type=str, default='log', help='the log name')
    parser.add_argument('--round', type=int, default=1, help='record the experiment')
    # parser.add_argument('--round', type=str, default="/home/liuxiangxi/WJY/zhaorongchen/baseline/OurModel/data/src", help='save dir for the embeddings')

    # params for transformer
    parser.add_argument('--lr1', type=float, default=0.0001, help='learning rate')#
    parser.add_argument('--seq_len', type=int, default=5, help='sequence length for input data')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model embedding')#
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--num_formers', type=int, default=3, help='number of transformer layers')#

    # params for diffusion
    parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')#
    parser.add_argument('--steps', type=int, default=5, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.0001)
    parser.add_argument('--noise_max', type=float, default=0.02)
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=10, help='steps for sampling/denoising')
    parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
    return parser

#############################################改evaluate为query预测的指标################################
# def evaluate(data_loader, data_te, mask_his, topN):
#     model.eval()
#     Autoencoder.eval()
#     e_idxlist = list(range(mask_his.shape[0]))
#     e_N = mask_his.shape[0]

#     predict_items = []
#     target_items = []
#     for i in range(e_N):
#         target_items.append(data_te[i, :].nonzero()[1].tolist())
    
#     if args.n_cate > 1:
#         category_map = Autoencoder.category_map.to(device)
    
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(data_loader):
#             batch = batch.to(device)

#             # mask map
#             his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]

#             _, batch_latent, _ = Autoencoder.Encode(batch)
#             batch_latent_recon = diffusion.p_sample(model, batch_latent, args.sampling_steps, args.sampling_noise)
#             prediction = Autoencoder.Decode(batch_latent_recon)  # [batch_size, n1_items + n2_items + n3_items]

#             prediction[his_data.nonzero()] = -np.inf  # mask ui pairs in train & validation set

#             _, mapped_indices = torch.topk(prediction, topN[-1])  # topk category idx

#             if args.n_cate > 1:
#                 indices = category_map[mapped_indices]
#             else:
#                 indices = mapped_indices

#             indices = indices.cpu().numpy().tolist()
#             predict_items.extend(indices)

#     test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

#     return test_results
#######################################################################################################


parser = argparse.ArgumentParser(description='')
parser = parse_global_args(parser)
parser = UniSAR.parse_model_args(parser)
parser = SarRunner.parse_runner_args(parser)
args, extras = parser.parse_known_args()
args = parser.parse_args()
print("args:", args)

if args.gpu == 'cpu':
    args.device = torch.device('cpu')
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.gpu != '' and torch.cuda.is_available():
        args.device = torch.device('cuda')

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))





if args.data == 'KuaiSAR':
    const.init_setting_KuaiSAR()
elif args.data == 'Amazon':
    const.init_setting_Amazon()
else:
    raise ValueError('Dataset Error')

utils.setup_seed(args.random_seed)
print('data ready.')
# train_path = args.data_path + 'train_list.npy'
# valid_path = args.data_path + 'valid_list.npy'
# test_path = args.data_path + 'test_list.npy'

# train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)
# train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
# test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

# if args.tst_w_val:
#     tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A))
#     test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
# mask_tv = train_data + valid_y_data

# print('data ready.')
### Build Autoencoder ###
# emb_path = args.emb_path + args.dataset + '/item_emb.npy'
# item_emb = torch.from_numpy(np.load(emb_path, allow_pickle=True))
# assert len(item_emb) == n_item
# out_dims = eval(args.out_dims)
# in_dims = eval(args.in_dims)[::-1]
# Autoencoder = AE(item_emb, args.n_cate, in_dims, out_dims, device, args.act_func, args.reparam).to(device)


# Build Pipeline ###
model: BaseModel = UniSAR(args)
model.load_model(model_path=args.test_path)
runner: BaseRunner = SarRunner(args)
###Build Transformer ###
transformer = former(d_model=args.d_model, n_heads=args.n_heads, num_layers=args.num_formers)
### Build Gaussian Diffusion ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)
diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, model.device).to(model.device)
transformer = transformer.to(model.device)

#########################################################改model为transformer#######################################################
### Build MLP ###
# latent_size = in_dims[-1]
# mlp_out_dims = eval(args.mlp_dims) + [latent_size]
# mlp_in_dims = mlp_out_dims[::-1]
# model = DNN(mlp_in_dims, mlp_out_dims, args.emb_size, time_type="cat", norm=args.norm, act_func=args.mlp_act_func).to(device)

# param_num = 0
# AE_num = sum([param.nelement() for param in Autoencoder.parameters()])
# mlp_num = sum([param.nelement() for param in model.parameters()])
# diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
# param_num = AE_num + mlp_num + diff_num
# print("Number of parameters:", param_num)

if args.optimizer == 'Adagrad':
    optimizer = optim.Adagrad(
        transformer.parameters(), lr=args.lr1, initial_accumulator_value=1e-8, weight_decay=args.wd)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(transformer.parameters(), lr=args.lr1, weight_decay=args.wd)
elif args.optimizer == 'AdamW':
    optimizer = optim.AdamW(transformer.parameters(), lr=args.lr1, weight_decay=args.wd)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(transformer.parameters(), lr=args.lr1, weight_decay=args.wd)
elif args.optimizer == 'Momentum':
    optimizer = optim.SGD(transformer.parameters(), lr=args.lr1, momentum=0.95, weight_decay=args.wd)

print("models ready.")
#########################################################################################################################################################









best_recall, best_epoch = -100, 0
best_test_result = None
update_count = 0
update_count_vae = 0
save_path = args.save_path + args.data + '/'

# if args.n_cate > 1:
#     start_time = time.time()
#     category_map = Autoencoder.category_map.cpu().numpy()
#     reverse_map = {category_map[i]:i for i in range(len(category_map))}
#     # mask for validation: train_dataset
#     mask_idx_train = list(train_data.nonzero())
#     mapped_mask_iid_train = np.array([reverse_map[mask_idx_train[1][i]] for i in range(len(mask_idx_train[0]))])
#     mask_train = sp.csr_matrix((np.ones_like(mask_idx_train[0]), (mask_idx_train[0], mapped_mask_iid_train)), \
#         dtype='float64', shape=(n_user, n_item))

#     # mask for test: train_dataset + valid_dataset
#     mask_idx_val = list(valid_y_data.nonzero())  # valid dataset
#     mapped_mask_iid_val = np.array([reverse_map[mask_idx_val[1][i]] for i in range(len(mask_idx_val[0]))])
#     mask_val = sp.csr_matrix((np.ones_like(mask_idx_val[0]), (mask_idx_val[0], mapped_mask_iid_val)), \
#         dtype='float64', shape=(n_user, n_item))

#     mask_tv = mask_train + mask_val

#     print("Preparing mask for validation & test costs " + time.strftime(
#                             "%H: %M: %S", time.gmtime(time.time()-start_time)))
# else:
#     mask_train = train_data

# print("Start training...")
train_loader = runner.src_train_loader
val_loader = runner.src_val_loader
test_loader = runner.src_test_loader
#################################################################################改训练文件#####################################################################
for epoch in range(1, args.epochs + 1):
    if epoch - best_epoch >= 20:
        print('-'*18)
        print('Exiting from training early')
        break

    transformer.train()
    model.eval()

    start_time = time.time()

    batch_count = 0
    total_loss = 0.0
    
    
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        # if epoch % 1 == 0:
        #     test_results,_ = runner.evaluate_diff(model, "test", diffusion, transformer,args.steps,args.sampling_noise)
        #     val_results,_ = runner.evaluate_diff(model, "val", diffusion, transformer,args.steps,args.sampling_noise)
        #     if val_results["rec"] > best_recall: # recall@20 as selection
        #         best_recall, best_epoch = val_results["rec"], epoch
        #         best_results = val_results
        #         best_test_results = test_results
        #         save_path = os.path.join(args.save_path, args.data)
        #         if not os.path.exists(save_path):
        #             os.makedirs(save_path)
        #         torch.save(transformer, '{}{}_{}lr1_{}wd1_{}wd2_bs{}_cate{}_in{}_out{}_lam{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_{}.pth' \
        #             .format(save_path, args.dataset, args.lr1, args.wd1, args.wd2, args.batch_size, args.n_cate, \
        #             args.in_dims, args.out_dims, args.lamda, args.mlp_dims, args.emb_size, args.mean_type, args.steps, \
        #             args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.log_name))

        #     print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
        #                         "%H: %M: %S", time.gmtime(time.time()-start_time)))
        #     print('---'*18)
        batch_count += 1
        optimizer.zero_grad()
        rec_fusion, src_fusion,query_emb = runner.get_batch_emb(model,batch)
        eq = query_emb
        vs = src_fusion
        vr = rec_fusion
        # vs = src_fusion.view(query_emb.size(0), query_emb.size(1), -1)
        # vr = rec_fusion.view(query_emb.size(0), query_emb.size(1), -1)
        # rec_fusion = rec_fusion.view(query_emb.size(0), query_emb.size(0), -1)
        # query_emb = query_emb.unsqueeze(1).expand(-1, src_fusion.size(1), -1)
        # print(type(query_emb),type(rec_fusion),type(src_fusion))
        # print(query_emb.shape,rec_fusion.shape,src_fusion.shape)
        eq = eq.to(model.device)
        vs = vs.to(model.device)
        vr = vr.to(model.device)
        # eq = utils.batch_to_gpu(query_emb,model.device)
        # vs = utils.batch_to_gpu(src_fusion,model.device)      
        terms = diffusion.training_losses(transformer, x_start=eq, v=vs, reweight=args.reweight)
        elbo = terms["loss"].mean()  # loss from diffusion
        batch_latent_recon = terms["pred_xstart"]
        
        loss = elbo
        
        update_count_vae += 1

        total_loss += loss
        loss.backward()
        optimizer.step()

    update_count += 1
    
    if epoch % 5 == 0:
        test_results,_ = runner.evaluate_diff(model, "test", diffusion, transformer,args.steps,args.sampling_noise)
        # val_results,_ = runner.evaluate_diff(model, "val", diffusion, transformer,args.steps,args.sampling_noise)
#         Traceback (most recent call last):
#   File "/home/liuxiangxi/WJY/zhaorongchen/baseline/OurModel/pretrain.py", line 344, in <module>
#     if val_results["rec"] > best_recall: # recall@20 as selection
# TypeError: '>' not supported between instances of 'str' and 'int'
        match = re.search(r'HR@20:(\d+\.\d+)', test_results["rec"])
        hr20 = float(match.group(1))
        print(test_results["rec"])
        if hr20 > best_recall: # recall@20 as selection
            best_recall, best_epoch = hr20, epoch
            best_results = test_results
            save_path = os.path.join(args.save_path, args.data)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(transformer, '{}{}_QueryGen.pth' \
                .format(save_path, args.data))

    print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))
    print('---'*18)

print('==='*18)
print("End. Best Epoch {:03d} ".format(best_epoch))
print(best_results) 
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))




