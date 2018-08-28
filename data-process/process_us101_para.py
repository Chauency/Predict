#%%

import pandas as pd
import numpy as np
from argparse import ArgumentParser
import multiprocessing as mp
import os
import sys
import time

#%% Settings

parser = ArgumentParser()
parser.add_argument('output', help='output dataset name (eg: pred1)')
parser.add_argument('-t', '--test', type=int,
                    help='consider only TEST number of vehicles')
parser.add_argument('-n', '--nproc', type=int, default=40,
                    help='number of processes (default: 40)')
parser.add_argument('-b', '--blocksize', type=int, default=2000,
                    help='size (pairs of vehicles) of each task block '+
                    '(default 2000)')
args = parser.parse_args()

# set valid range
vr = 300  # about 100 meters

# set valid sequence length
isl = 10  # input sequence length
delay = 1  # predict delay (s)
dsl = round(delay * 10)  # delay sequence length
tsl = isl + dsl  # total sequence length

# set features to use
# ignore v_Vel and recompute velocity, ignore lanes
cols = ['Vehicle_ID', 'Global_Time', 'Local_X', 'Local_Y']

# data directory
dd = '/home/alan/work/data/'

#%%

print()

if args.test:
    print('Running in TEST mode.')
    print(f'Vehicle number: {args.test}')
else:
    print('Running in NORMAL mode')

print(f'Process number: {args.nproc}')
print(f'Task block size: {args.blocksize}')

if os.path.exists(f'{dd}{args.output}.csv'):
    print(f'WARNING: file "{dd}{args.output}.csv" already exists')

print()
if input('Continue? [y/n] ') != 'y':
    sys.exit('Program terminated\n')

#%% Define helper function for multiprocessing


def helper1(data, qi, qo, counter1):
    dic = {}
    while True:
        vehs = qi.get()

        if vehs is None:
            break

        for veh in vehs:
            traj = data[data.Vehicle_ID == veh].drop(columns='Vehicle_ID')
            traj = traj.sort_values('Global_Time').set_index('Global_Time')

            # split and compute velocity
            times = traj.index.values
            gaps = times[1:] - times[:-1] - 100
            nodes = gaps.nonzero()[0] + 1
            nodes = [0] + nodes.tolist() + [len(times)]
            pieces = []
            for i in range(len(nodes) - 1):
                piece = traj[nodes[i]:nodes[i+1]]
                x = piece.Local_X.values
                vx = (x[1:] - x[:-1]) / 0.1
                y = piece.Local_Y.values
                vy = (y[1:] - y[:-1]) / 0.1
                piece = piece[1:].assign(Vel_X=vx, Vel_Y=vy)
                pieces.append(piece)

            # combine and store
            dic[veh] = pd.concat(pieces)
            with counter1.get_lock():
                counter1.value += 1

    # return dic
    qo.put(dic)


def helper2(df, dic, vr, tsl, isl, q, counter2, lock, ID):
    nobs = 0  # count obs in this process
    while True:
        pairs = q.get()

        if pairs is None:
            break

        for pair in pairs:
            e = dic[pair[0]]  # ego
            t = dic[pair[1]]  # target

            # find valid times
            synTime = e.index.intersection(t.index)
            esyn = e.loc[synTime]
            tsyn = t.loc[synTime]
            xrela = tsyn.Local_X - esyn.Local_X
            yrela = tsyn.Local_Y - esyn.Local_Y
            drela = np.sqrt(xrela**2 + yrela**2)
            valid = synTime[drela <= vr]

            # generate examples and append
            rela = pd.DataFrame({'Rela_X': xrela,
                                 'Rela_Y': yrela,
                                 'Ego_Vel_X': esyn.Vel_X,
                                 'Ego_Vel_Y': esyn.Vel_Y,
                                 'Target_Vel_X': tsyn.Vel_X,
                                 'Target_Vel_Y': tsyn.Vel_Y})
            rela = rela.loc[valid]
            for i in range(0, len(rela) - tsl, tsl):  # step tsl every time
                if (rela.index[i] + (tsl - 1)*100) == rela.index[i + tsl - 1]:
                    nobs += 1
                    obs = rela.iloc[list(range(i, i + isl)) + [i + tsl - 1]]
                    obs = obs.reset_index(drop=True)
                    obs.insert(0, 'Tag',
                               [f'{ID}-{nobs}-i']*isl + [f'{ID}-{nobs}-t'])
                    df = df.append(obs)

            # count
            with counter2.get_lock():
                counter2.value += 1

    # save dataset with lock
    with lock:
        df.to_csv(f'{dd}{args.output}.csv', mode='a',
                  index=False, header=False)


#%% Start timing

start = time.time()

#%% Process us-101 dataset to generate dictionary for later use

# read us-101.csv
print()
print('Reading us-101 dataset...')
data = pd.read_csv(f'{dd}us-101.csv', usecols=cols)
print('Done')

# shift time to use smaller values
data.Global_Time -= data.Global_Time.min()

# generate dicts of each vehicle's trajectory
print()
print('Generating dicts...')
vehs = data.Vehicle_ID.unique()
if args.test:
    vehs = vehs[:args.test]
nvehs = len(vehs)  # number of vehicles
bs = 5  # block size
nb = nvehs // bs + 1 # number of blocks
qi = mp.Queue()  # input queue
for i in range(nb):
    qi.put(vehs[i*bs:(i+1)*bs])
for i in range(args.nproc):
    qi.put(None)
qo = mp.Queue()
counter1 = mp.Value('i', 0)
procs1 = []
for i in range(args.nproc):
    p = mp.Process(target=helper1, args=(data, qi, qo, counter1))
    p.start()
    procs1.append(p)

# report
while counter1.value < nvehs:
    print(f'\r[{counter1.value}/{nvehs} vehicles processed]', end='')
print(f'\r[{counter1.value}/{nvehs} vehicles processed]')

# combine dicts
print('Combining dicts...')
dic = {}
for i in range(args.nproc):
    dic.update(qo.get())

for p in procs1:
    p.join()

print('Done')

#%% Generate and save dataset with multiprocessing

print()
print('Generating and saving dataset...')

# save header first
df = pd.DataFrame(columns=['Tag', 'Rela_X', 'Rela_Y', 'Ego_Vel_X',
                           'Ego_Vel_Y', 'Target_Vel_X', 'Target_Vel_Y'])
df.to_csv(f'{dd}{args.output}.csv', index=False)

# process pairs of vehicles with multiprocessing
pairs = [(i, j) for i in vehs for j in vehs if i < j]
npair = len(pairs)
nblock = npair // args.blocksize + 1  # number of blocks
q = mp.Queue()
for i in range(nblock):
    q.put(pairs[i*args.blocksize:(i+1)*args.blocksize])
for i in range(args.nproc):
    q.put(None)
counter2 = mp.Value('i', 0)
procs2 = []
lock = mp.Lock()
for i in range(args.nproc):
    p = mp.Process(target=helper2,
                   args=(df, dic, vr, tsl, isl, q, counter2, lock, i))
    p.start()
    procs2.append(p)

# report
while True:
    status = [p.is_alive() for p in procs2]
    print(f'\r[{counter2.value}/{npair} vehicle pairs processed]',
          f'[{args.nproc - sum(status)}/{args.nproc} processes completed]',
          end='')
    if sum(status) == 0:
        print()
        break

for p in procs2:
    p.join()

print('Done')

#%% Report time used

end = time.time()
last = time.strftime('%Hh %Mm %Ss', time.gmtime(round(end - start)))
print()
print(f'Time used: {last}')
print()
