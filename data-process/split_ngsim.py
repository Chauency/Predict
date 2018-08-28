#%%

import pandas as pd

#%%

print('Reading NGSIM dataset...')
ngsim = pd.read_csv('/home/alan/work/data/NGSIM.csv')
names = ngsim.Location.dropna().unique()
for name in names:
    print(f'Saving {name} dataset...')
    data = ngsim[ngsim.Location == name]
    data.to_csv(f'/home/alan/work/data/{name}.csv', index=False)
print('Finish')

#%% fail

#location = pd.read_csv('~/data/NGSIM.csv', usecols=['Location'], squeeze=True)
#names = location.drop_duplicates().dropna().tolist()
#for name in names:
#    print('Seperating dataset %s...' % name)
#    skip = location.index[location != name]
#    data = pd.read_csv('~/data/NGSIM.csv', skiprows=skip)
#    print('Saving...')
#    data.to_csv('~/data/%s.csv' % name)
#    print('Finish')
#    print()

#%%

#count = 0
#for name in names:
#    tmp = location == name
#    count += tmp.sum()
#print(count)
#print(len(location))

#%% index of nan

#for i in location.index:
#    if location[i] not in names:
#        print(i)