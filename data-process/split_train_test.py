#%%

from argparse import ArgumentParser
import pandas as pd
import os
import sys
import random

#%%

parser = ArgumentParser()
parser.add_argument('target', help='dataset to split (eg: pred1)')
parser.add_argument('-t', '--test', type=int,
                    help='use TEST observations from target dataset')
parser.add_argument('-p', '--proportion', type=float, default=0.3,
                    help='proportion of test set (default: 0.3)')
parser.add_argument('-r', '--random', action='store_true',
                    help='shuffle randomly before splitting')
parser.add_argument('-y', '--yes', action='store_true',
                    help='answer "yes" to all prompts')
args = parser.parse_args()

dd = '/home/alan/work/data/'
nrow = 11

#%%

def stop():
    sys.exit('Program terminated\n')

def check():
    print()
    if (not args.yes) and (input('Continue? [y/n] ') != 'y'):
        stop()


#%% Check args

print()

if args.test:
    print('Running in TEST mode')
    print(f'Using {args.test} observations')
else:
    print('Running in NORMAL mode')
print(f'Target dataset: {args.target}')
print(f'Proportion of test set: {args.proportion}')
print(f'Random: {args.random}')
if os.path.exists(f'{dd}{args.target}_train.csv'):
    print(f'WARNING: file "{dd}{args.target}_train.csv" exists')
if os.path.exists(f'{dd}{args.target}_test.csv'):
    print(f'WARNIGN: file "{dd}{args.target}_test.csv" exists')
check()

#%%

print()
print('Reading dataset...')
if args.test:
    data = pd.read_csv(f'{dd}{args.target}.csv', nrows=nrow*args.test)
else:
    data = pd.read_csv(f'{dd}{args.target}.csv')

if len(data) % nrow != 0:
    print('WARNING: incorrect number of rows in target dataset')
    check()

print('Done')

#%%

print()
print('Splitting dataset...')
nobs = len(data) // nrow
idx = list(range(nobs))
if args.random:
    random.shuffle(idx)
split = round(nobs * args.proportion)

# test set
test = [j for i in idx[:split] for j in range(i*nrow,(i+1)*nrow)]
test = data.iloc[test]

# training set
train = [j for i in idx[split:] for j in range(i*nrow,(i+1)*nrow)]
train = data.iloc[train]

print('Done')

#%%

print()
print('Saving training and test sets...')
train.to_csv(f'{dd}{args.target}_train.csv', index=False)
print('Training set saved')
test.to_csv(f'{dd}{args.target}_test.csv', index=False)
print('Test set saved')
print('Done')

print()
