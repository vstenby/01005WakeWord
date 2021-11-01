import pandas as pd

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

    counts['percentage'] = counts['n'].cumsum() / counts['n'].sum()
    
    train_ID = (counts['ID'].loc[counts['percentage'] <= 0.7]).tolist()
    val_ID   = (counts['ID'].loc[(counts['percentage'] > 0.7)&(counts['percentage']<= 0.85)]).tolist()
    test_ID  = (counts['ID'].loc[counts['percentage'] > 0.85]).tolist()
    
    train = data.loc[data['ID'].isin(train_ID)].sort_values(by=['ID', 't1'])
    val   = data.loc[data['ID'].isin(val_ID)].sort_values(by=['ID', 't1'])
    test  = data.loc[data['ID'].isin(test_ID)].sort_values(by=['ID', 't1'])
    
    return train, val, test