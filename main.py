from argparse import ArgumentParser
import multiprocessing as mp
import time
import os
import sys
sys.path.append('/home/alan/work/pred/')

import torch
from torch.utils.data import DataLoader

from datasets import Pred1
import models
import misc

#-----------------------------------------------------------------------------

# parse arguments
parser = ArgumentParser()
parser.add_argument('-m', '--model', default='Model1',
                    help='name of model to use (default: Model1)')
parser.add_argument('-l', '--load',
                    help='load the given model and continue training')
parser.add_argument('-d', '--dataset', default='pred1_20',
                    help='name of dataset to use (default: pred1_20)')
parser.add_argument('-s', '--subset', type=int,
                    help='use SUBSET obs from training set')
parser.add_argument('-e', '--num-epochs', type=int, default=1,
                    help='number of training epochs (default: 1)')
parser.add_argument('-w', '--num-workers', type=int, default=4,
                    help='number of workers for loading dataset '
                    +'(default: 4)')
parser.add_argument('-b', '--batch-size', type=int, default=40,
                    help='batch size (default: 40)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate of the optimizer '
                    +'(default: 0.001)')
parser.add_argument('--lmd', type=float,
                    help='lambda of the regularization term')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='maximum gradient norm (default: 50)')
parser.add_argument('--path', default='/home/alan/work/pred/saved/',
                    help='path to save output files '
                    +'(default: /home/alan/work/pred/saved/)')
args = parser.parse_args()

# figure out output settings
date = time.strftime('%m%d', time.gmtime())
trials = [0]
for f in os.listdir(args.path):
    if f.startswith(date):
        trials.append(int(f.split('-')[1]))
trial = max(trials) + 1
output_path = args.path + f'{date}-{trial}/'
prefix = f'{date}-{trial}-'  # output files prefix

# figure out load settings
if args.load:
    lst = args.load.split('-')
    load_path = f'{args.path}{lst[0]}-{lst[1]}/{args.load}.pth'

# set up GPU
ngpu = torch.cuda.device_count()
device = torch.device('cuda:0' if ngpu > 0 else 'cpu')

model = getattr(models, args.model)  # fetch model

#-----------------------------------------------------------------------------

# print settings information
print()
settings = []
if args.load:
    settings.append(f'Continue training from: {args.load}.pth')
else:
    settings.append(f'Fresh start')
settings.append(f'Model: {args.model}')
settings.append(f'Dataset: {args.dataset}'+
                (f' (using SUBSET of {args.subset} obs)'
                 if args.subset else ''))
settings.append(f'Using GPU: {False if ngpu == 0 else f"{ngpu} GPU"}')
settings.append(f'Number of training epochs: {args.num_epochs}')
settings.append(f'Number of workers: {args.num_workers}')
settings.append(f'Batch size: {args.batch_size}')
settings.append(f'Learning rate: {args.lr}')
settings.append(f'Lambda of the regularization term: {args.lmd}')
settings.append(f'Output path: {output_path}')
report = '\n'.join(settings)
print(report)

# check with prompt
print()
if input('Continue? [y/n] ') != 'y':
    sys.exit('Program terminated\n')

os.makedirs(output_path, exist_ok=True)  # make sure output path exists

# create log
with open(f'{output_path}log.txt', mode='w') as log:
    log.write(report)

#-----------------------------------------------------------------------------

print()
print('Preparing...')

# training data
train_data = Pred1(args.dataset, subset=args.subset)
train_m = len(train_data)  # number of observations in the training set
train_loader = DataLoader(train_data, batch_size=args.batch_size,
                          num_workers=args.num_workers, shuffle=True,
                          pin_memory=(True if ngpu > 0 else False))
train_num_batches = len(train_loader)  # number of batches of training

# test data
test_data = Pred1(args.dataset, train=False)
test_m = len(test_data)  # number of observations in the test set
test_loader = DataLoader(test_data, batch_size=args.batch_size,
                         num_workers=args.num_workers,
                         pin_memory=(True if ngpu > 0 else False))
test_num_batches = len(test_loader)  # number of batches of test

# start helper process
q = mp.Queue()
p = mp.Process(target=misc.test_error_writer,
               args=(f'{output_path}{prefix}test_error',
                     args.num_epochs, test_num_batches, q))
p.start()

# training model
train_net = model(6, 2)
train_net.double()
if ngpu > 1:
    train_net = torch.nn.DataParallel(train_net)
train_net.to(device)
if args.load:
    train_net.load_state_dict(torch.load(load_path))
else:
    train_net.apply(misc.weights_init)

# test model
test_net = model(6, 2)
test_net.double()
if ngpu > 1:
    test_net = torch.nn.DataParallel(test_net)
for param in test_net.parameters():
    param.requires_grad = False
test_net.to(device)

# optimizer and criterion
optimizer = torch.optim.Adam(train_net.parameters(), lr=args.lr)
train_criterion = torch.nn.MSELoss(reduction='sum')
train_criterion.to(device)
test_criterion = torch.nn.MSELoss(reduction='none')
test_criterion.to(device)

h1, c1, h2, c2 = [torch.empty(1, dtype=torch.float64, device=device)
                  for i in range(4)]  # initial states of lstm

# write headers
with open(f'{output_path}{prefix}test_error.csv', mode='w') as f:
    f.write('Test_Err\n')
with open(f'{output_path}{prefix}train_error.csv', mode='w') as f:
    f.write('Train_Err\n')
print('Done')

#-----------------------------------------------------------------------------

print()
print('Training...')

# train then test the model for each epoch
for epoch in range(args.num_epochs):
    print(f'Epoch {epoch+1}')

    start = time.time()  # time training
    train_error = 0  # cumulate training error of each epoch
    for i, sample in enumerate(train_loader):
        sequence, target = sample
        batch_size = target.shape[0]

        # prepare input
        sequence = sequence.to(device)
        h1 = h1.new_zeros(batch_size, 128)
        c1 = c1.new_zeros(batch_size, 128)
        h2 = h2.new_zeros(batch_size, 128)
        c2 = c2.new_zeros(batch_size, 128)
        inputs = (sequence, (h1, c1), (h2, c2))

        # forward and compute loss
        output = train_net(inputs)
        target = target.to(device)
        loss = 0.5 * train_criterion(output, target)
        if args.lmd:
            reg = args.lmd * misc.reg_l2_fc(train_net)  # l2 regularization
        else:
            reg = 0

        # step
        optimizer.zero_grad()
        (loss/batch_size + reg).backward()  # backprop total loss
        torch.nn.utils.clip_grad_norm_(train_net.parameters(),
                                       args.max_grad_norm)
        optimizer.step()

        train_error += float(loss)  # cumulate training error
        print(f'\rtrain: [{i+1}/{train_num_batches} batch]', end='')

    print(f' [{(time.time()-start):.2f}s]')  # report time used by training
    with open(f'{output_path}{prefix}train_error.csv', mode='a') as f:
        f.write(f'{train_error/train_m}\n')  # write training error
    state = train_net.state_dict()
    torch.save(state, f'{output_path}{prefix}{epoch+1}.pth')  # store model

    #-------------------------------------------------------------------------

    test_net.load_state_dict(state)
    start = time.time()  # time test
    test_error = 0  # cumulate test error of each epoch
    for i, sample in enumerate(test_loader):
        sequence, target = sample
        batch_size = target.shape[0]

        # prepare input
        sequence = sequence.to(device)
        h1 = h1.new_zeros(batch_size, 128)
        c1 = c1.new_zeros(batch_size, 128)
        h2 = h2.new_zeros(batch_size, 128)
        c2 = c2.new_zeros(batch_size, 128)
        inputs = (sequence, (h1, c1), (h2, c2))

        # forward and compute loss
        output = test_net(inputs)
        target = target.to(device)
        loss = torch.sum(test_criterion(output, target), 1)
        loss = loss.cpu().numpy()
        q.put(loss)
        test_error += loss.sum()
        print(f'\rtest: [{i+1}/{test_num_batches} batch]', end='')

    print(f' [{(time.time()-start):.2f}s]')  # report time used by test
    with open(f'{output_path}{prefix}test_error.csv', mode='a') as f:
        f.write(f'{test_error/test_m}\n')  # write test error

    #-------------------------------------------------------------------------

p.join()
print('Done')
print()
