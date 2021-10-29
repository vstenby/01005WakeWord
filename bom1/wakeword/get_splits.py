import pandas as pd

def get_splits():
    '''
    Returns 
        train, test, val splits
    '''
    
    links = pd.read_csv('./csv/links.csv')
    data  = pd.read_csv('./csv/data.csv')

    counts = pd.merge(links[['semester', 'title', 'ID']],
                  data.groupby(['semester', 'ID']).size().reset_index(name='n'),
                  how='inner',
                  on=['semester', 'ID'])

    counts['percentage'] = counts['n'].cumsum() / counts['n'].sum()
    
    train_ID = (counts['ID'].loc[counts['percentage'] <= 0.7]).tolist()
    val_ID   = (counts['ID'].loc[(counts['percentage'] > 0.7)&(counts['percentage']<= 0.85)]).tolist()
    test_ID  = (counts['ID'].loc[counts['percentage'] > 0.85]).tolist()
    
    train = data.loc[data['ID'].isin(train_ID)].sort_values(by=['ID', 't1'])
    val   = data.loc[data['ID'].isin(val_ID)].sort_values(by=['ID', 't1'])
    test  = data.loc[data['ID'].isin(test_ID)].sort_values(by=['ID', 't1'])
    
    return train, val, test

def get_balanced_splits(balance, seed=42):
    '''
    Get balanced splits. 
    Balance can either be 
           "1:x", where x is the number of class0.
           "everyx", where every x'th t1 is taken. Since ffmpeg increments are 0.1 seconds, then balance=every50 means that t1 = [0, 0.5, ...]
           "all" meaning that we take all negative classes. This is a lot of negatives.
    
    They are excluded based on the fact that they are not allowed to overlap. 
    '''
       
    #Fetch the durations.
    durations = lecture_durations()

    np.random.seed(seed)

    assert (':' in balance) or (balance == 'all') or ('every' in balance), 'args.balance should either be 1:x where x is the number of class 0 sampled from each lecture, or "all"'

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
    train, val, test = get_splits()
    
    modified_sets = []
    
    for split in ['train', 'val', 'test']:
        splitname    = split
        split        = eval(split) #Load the dataframe we're talking about.

        #Get the counts for each ID.
        counts = dict(split.groupby(['ID']).size())

        #Preallocate a dataset for the negative classes.
        class0 = pd.DataFrame(columns = ['ID', 't1', 't2'])

        for ID in tqdm.tqdm(split['ID'].unique(), desc=f'Generating new clips for {splitname}'):

            #Fetch all t.
            t_linspace  = np.arange(1, durations[ID] - 1, 0.01) #This is the finest grid possible with our download.

            #Fetch the t from wakewords.
            t_wakewords = split[['t1', 't2']].loc[split['ID'] == ID].mean(axis=1).tolist()

            for t in t_wakewords:
                #Remove linspace with overlap. Here, we remove ones with -any- overlap at all.
                t_linspace = t_linspace[1 <= np.abs((t_linspace - t))]

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

            t1 = t_selected - 1
            t2 = t_selected + 1

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