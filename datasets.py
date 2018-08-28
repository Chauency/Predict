import pandas as pd
from torch.utils.data import Dataset


DD = '/home/alan/work/data/'  # data directory


class Pred1(Dataset):

    def __init__(self, name, train=True, subset=None, dd=DD):
        self.nrow = 11  # number of rows in one obs

        if train:
            if subset:
                self.df = pd.read_csv(f'{dd}{name}_train.csv',
                                      nrows=self.nrow*subset)
            else:
                self.df = pd.read_csv(f'{dd}{name}_train.csv')
        else:
            if subset:
                self.df = pd.read_csv(f'{dd}{name}_test.csv',
                                      nrows=self.nrow*subset)
            else:
                self.df = pd.read_csv(f'{dd}{name}_test.csv')

        if len(self.df) % self.nrow != 0:
            print('\nWarning: incorrect number of rows in dataframe')

        if 'Tag' in self.df.columns:  # Tag is for debugging
            self.df.drop('Tag', axis=1, inplace=True)

        self.nobs = len(self.df) // self.nrow
        self.name = f'{name}_train'

    def __len__(self):
        return self.nobs

    def __getitem__(self, idx):
        obs = self.df.iloc[idx*self.nrow : (idx+1)*self.nrow]
        sequence = obs.iloc[:-1].values
        target = obs[['Rela_X', 'Rela_Y']].iloc[-1].values
        return (sequence, target)

    def __str__(self):
        return self.name