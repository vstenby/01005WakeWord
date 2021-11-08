import pandas as pd
from .lecture_durations import lecture_durations
import torchaudio
import numpy as np

def get_splits(cliplength = 2):
    '''
    Returns 
        train, test, val splits
    '''
    
    halfclip = cliplength/2
    
    links = pd.read_csv('./csv/links.csv')
    data  = pd.read_csv('./csv/data.csv')
    
    data['t1'] = data['t'] - halfclip
    data['t2'] = data['t'] + halfclip
    data.drop('t', axis=1, inplace=True)

    counts = pd.merge(links[['semester', 'title', 'ID']],
                  data.groupby(['semester', 'ID']).size().reset_index(name='n'),
                  how='inner',
                  on=['semester', 'ID'])

    #Remove IDs where duration according to video.dtu.dk and according to torchaudio doesn't match. This seems to be a problem for 12 of the lectures... Not sure why. Perhaps downloading the lectures again will fix the issue.
    def d(ID):
        info = torchaudio.info(f'/work3/s164419/01005WakeWordData/lectures/{ID}.wav')
        return info.num_frames / info.sample_rate
    
    durations = lecture_durations()
    
    #This is a hardcode fix and will probably needed to be fixed later on.
    counts['d1'] = counts['ID'].apply(lambda x : durations[x])
    counts['d2'] = counts['ID'].apply(lambda x : d(x))

    counts = counts.loc[np.abs(counts['d2'] - counts['d1']) < 1] #This will remove 11 of the lectures that caused problems.
    
    counts['percentage'] = counts['n'].cumsum() / counts['n'].sum()
    
    
    
    
    
    
    
    
    
    
    train_ID = (counts['ID'].loc[counts['percentage'] <= 0.7]).tolist()
    val_ID   = (counts['ID'].loc[(counts['percentage'] > 0.7)&(counts['percentage']<= 0.85)]).tolist()
    test_ID  = (counts['ID'].loc[counts['percentage'] > 0.85]).tolist()
    
    train = data.loc[data['ID'].isin(train_ID)].sort_values(by=['ID', 't1'])
    val   = data.loc[data['ID'].isin(val_ID)].sort_values(by=['ID', 't1'])
    test  = data.loc[data['ID'].isin(test_ID)].sort_values(by=['ID', 't1'])
    
    return train, val, test