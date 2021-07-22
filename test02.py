from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime


import torch.optim as optim
from torch.optim import lr_scheduler

from utils_ar import *
from modules_pyt import *

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--num-atoms', type=int, default=5,
                    help='Number of atoms in simulation.')
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='_springs5',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='/home/sw/logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=4,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=49,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')
parser.add_argument('--loss_type',type=str,default='mse',
                    help='')
parser.add_argument('--node-hidden',type=int, default=64,
                    help='output node hidden')
parser.add_argument('--edge-hidden',type=int,default=64,
                    help='output edge hidden')
parser.add_argument('--global-hidden',type=int, default=64,
                    help='output global hidden')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    aargnn_file = os.path.join(save_folder, 'aargnn.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

train_loader, valid_loader, test_loader = getgraphattr(args.batch_size)

# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)

rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

#model paremeter
aargnn = RRGCN(n_in_node=args.dims,
               edge_types=args.edge_types,
               n_hid=args.decoder_hidden,
               node_hidden=args.node_hidden,
               edge_hidden=args.edge_hidden,
               global_hidden=args.global_hidden,
               do_prob=args.decoder_dropout)

definedloss = torch.nn.L1Loss()

if args.load_folder:
    aargnn_file = os.path.join(args.load_folder, 'aargnn.pt')
    aargnn.load_state_dict(torch.load(aargnn_file))
    args.save_folder = False

optimizer = optim.Adam(list(aargnn.parameters()),lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

if args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # TODO: hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)

    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    aargnn.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)


def train(epoch, best_val_loss):
    t = time.time()
    all_train = []
    rmse_train = []
    mae_train = []
    mape_train = []

    aargnn.train()
    scheduler.step()
    for batch_idx, (node_attr,edge_attr,global_attr,target_attr) in enumerate(train_loader):
        if args.cuda:
            node_attr, edge_attr, global_attr, target_attr = node_attr.cuda(),edge_attr.cuda(),global_attr.cuda(),target_attr.cuda()
        node_attr, edge_attr, global_attr, target_attr = Variable(node_attr,volatile=True), Variable(edge_attr,volatile=True), Variable(global_attr,volatile=True), Variable(target_attr,volatile=True)

        optimizer.zero_grad()

        output = aargnn(node_attr, edge_attr, global_attr, rel_rec, rel_send)

        target = target_attr
        predict_loss = definedloss(output, target)
        loss = predict_loss

        loss.backward()
        optimizer.step()

        all_train.append(loss.data[0])
        rmse_train.append(torch.sqrt(F.mse_loss(output, target).data[0]))
        mae_train.append(torch.nn.L1Loss(output, target).data[0])
        mask=target!=0
        mape_train.append(np.fabs((target[mask]-output[mask])/target[mask]).mean())


    all_val = []
    rmse_val = []
    mae_val = []
    mape_val = []

    aargnn.eval()

    for batch_idx, (node_attr,edge_attr,global_attr,target_attr) in enumerate(valid_loader):
        if args.cuda:
            node_attr, edge_attr, global_attr, target_attr = node_attr.cuda(), edge_attr.cuda(), global_attr.cuda(), target_attr.cuda()
        node_attr, edge_attr, global_attr, target_attr = Variable(node_attr), Variable(edge_attr), Variable(global_attr), Variable(target_attr)

        output = aargnn(node_attr, edge_attr, global_attr)

        target = target_attr
        loss = definedloss(output, target, args.var)

        all_val.append(loss.data[0])
        rmse_val.append(torch.sqrt(F.mse_loss(output, target).data[0]))
        mae_val.append(torch.nn.L1Loss(output, target).data[0])
        mask_val = target != 0
        mape_val.append(np.fabs((target[mask_val] - output[mask_val]) / target[mask_val]).mean())

    print('Epoch: {:04d}'.format(epoch),
          'rmse_train: {:.10f}'.format(np.mean(rmse_train)),
          'rmse_val: {:.10f}'.format(np.mean(rmse_val)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(rmse_val) < best_val_loss:
        torch.save(aargnn.state_dict(), aargnn_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'mse_train: {:.10f}'.format(np.mean(rmse_train)),
              'mse_val: {:.10f}'.format(np.mean(rmse_val)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return np.mean(rmse_val)


def test():
    all_test = []
    rmse_test = []
    mae_test = []
    mape_test = []
    tot_mse = 0
    counter = 0

    aargnn.eval()
    aargnn.load_state_dict(torch.load(aargnn_file))
    for batch_idx, (node_attr, edge_attr, global_attr,target_attr) in enumerate(test_loader):
        if args.cuda:
            node_attr, edge_attr, global_attr, target_attr = node_attr.cuda(), edge_attr.cuda(), global_attr.cuda(), target_attr.cuda()
        node_attr, edge_attr, global_attr, target_attr = Variable(node_attr), Variable(edge_attr), Variable(
                global_attr), Variable(target_attr)

        assert (node_attr.size(2) - args.timesteps) >= args.timesteps

        output = aargnn(node_attr, edge_attr, global_attr)

        target = target_attr
        loss = definedloss(output, target, args.var)

        all_test.append(loss.data[0])
        rmse_test.append(torch.sqrt(F.mse_loss(output, target).data[0]))
        mae_test.append(torch.nn.L1Loss(output, target).data[0])
        mask_test = target != 0
        mape_test.append(np.fabs((target[mask_test] - output[mask_test]) / target[mask_test]).mean())

        # For plotting purposes
        output=aargnn(node_attr, edge_attr, global_attr)

        mse = ((target - output) ** 2).mean(dim=0).mean(dim=0).mean(dim=-1)
        tot_mse += mse.data.cpu().numpy()
        counter += 1

    mean_mse = tot_mse / counter
    mse_str = '['
    for mse_step in mean_mse[:-1]:
        mse_str += " {:.12f} ,".format(mse_step)
    mse_str += " {:.12f} ".format(mean_mse[-1])
    mse_str += ']'

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('all_test: {:.10f}'.format(np.mean(all_test)),
          'mae_test: {:.10f}'.format(np.mean(mae_test)),
          'rmse_test: {:.10f}'.format(np.mean(rmse_test)),
          'mape_test: {:.10f}'.format(np.mean(mape_test)))
    print('MSE: {}'.format(mse_str))
    if args.save_folder:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print('nll_test: {:.10f}'.format(np.mean(nll_test)),
              'kl_test: {:.10f}'.format(np.mean(kl_test)),
              'mse_test: {:.10f}'.format(np.mean(mse_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)),
              file=log)
        print('MSE: {}'.format(mse_str), file=log)
        log.flush()


# Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    val_loss = train(epoch, best_val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()

test()
if log is not None:
    print(save_folder)
    log.close()

