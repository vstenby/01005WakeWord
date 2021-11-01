import pandas as pd
import numpy as np
import tqdm as tqdm
from .lecture_durations import lecture_durations
from .get_splits import get_splits

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
                t_linspace = t_linspace[halfclip <= np.abs((t_linspace - t))] #

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