import pandas as pd
from .lecture_durations import lecture_durations
import torchaudio
import numpy as np
import tqdm as tqdm
import os

def get_splits(local_path = '/zhome/55/f/127565/Desktop/01005WakeWord/', cliplength = 1, balance=None, seed=42, splits=['train', 'val', 'test']):
    '''
    Returns 
        train, test, val splits
    '''
    
    halfclip = cliplength/2
    
    links_path = local_path + 'csv/links.csv'
    data_path  = local_path + 'csv/data.csv'
    
    links = pd.read_csv(links_path)
    data  = pd.read_csv(data_path)
    
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

    if balance is not None:
        return get_balanced_splits(balance, seed, splits, cliplength)
    
    return train, val, test

def get_balanced_splits(balance, seed=42, splits = ['train', 'val', 'test'], cliplength=2):
    '''
    Get balanced splits. 
    Balance can either be 
           "1:x", where x is the number of class0.
           "everyx", where every x'th t1 is taken. Since ffmpeg increments are 0.1 seconds, then balance=every50 means that t1 = [0, 0.5, ...]
           "all" meaning that we take all negative classes. This is a lot of negatives.
    
    They are excluded based on the fact that they are not allowed to overlap. 
    
    Cliplength is the length of the clip in seconds.
    '''
    halfclip = cliplength/2
    
    #Fetch the durations.
    durations = lecture_durations()

    np.random.seed(seed)

    assert (':' in balance) or (balance == 'all') or ('every' in balance), 'args.balance should either be 1:x where x is the number of class 0 sampled from each lecture, everyn where n is e.g. every 5th clip, or "all"'

    if ':' in balance:
        _, ratio = balance.split(':')
        ratio = int(ratio)
    elif 'every' in balance:
        #Take every x'th
        every = int(balance.replace('every',''))
        ratio = None
    else:
        every = None
        ratio = None

    #Get train, val and test split to overwrite with negative classes.
    train, val, test = get_splits(cliplength=cliplength)
    
    modified_sets = []
    
    for split in ['train', 'val', 'test']:
        splitname    = split
        
        if split not in splits:
            #Just append None and skip the entire thing.
            modified_sets.append(None)
            continue
        
        split        = eval(split) #Load the dataframe we're talking about.

        #Get the counts for each ID.
        counts = dict(split.groupby(['ID']).size())

        #Preallocate a dataset for the negative classes.
        class0 = pd.DataFrame(columns = ['ID', 't1', 't2'])

        for ID in tqdm.tqdm(split['ID'].unique(), desc=f'Generating negative classes for {splitname}'):

            #Fetch all t.
            t_linspace  = np.arange(halfclip, durations[ID] - halfclip, 0.01) #This is the finest grid possible with our download.
            
                                
            #Fetch the t from wakewords.
            t_wakewords = split[['t1', 't2']].loc[split['ID'] == ID].mean(axis=1).tolist()

            for t in t_wakewords:
                #Remove linspace with overlap. Here, we remove ones with -any- overlap at all.
                t_linspace = t_linspace[cliplength <= np.abs((t_linspace - t))] #

            if ratio is not None:
                n_class0 = np.round(ratio * counts[ID])
                t_selected = np.random.choice(t_linspace, n_class0)
                t_selected = t_selected[np.argsort(t_selected)]
            elif every is not None:
                t_selected = t_linspace
                t_selected = t_selected[np.argsort(t_selected)]
                t_selected = t_selected[0:-1:every]
            else:
                #Then we just take all of them.
                t_selected = t_linspace[np.argsort(t_linspace)]

            t1 = t_selected - halfclip 
            t2 = t_selected + halfclip

            #Fill out the dataframe.
            class0_ID = pd.DataFrame(columns = ['ID', 't1', 't2'])
            class0_ID['t1'] = t1
            class0_ID['t2'] = t2
            class0_ID['ID'] = ID


            class0 = pd.concat([class0, class0_ID])

        #Now, we want to combine our class1 and class0.
        class1 = split[['ID', 't1', 't2']].copy()

        #Set the targets
        class0['class'] = 0
        class1['class'] = 1

        split = pd.concat([class0, class1])

        #Append the modified set.
        modified_sets.append(split)

    train, val, test = modified_sets
    return train, val, test