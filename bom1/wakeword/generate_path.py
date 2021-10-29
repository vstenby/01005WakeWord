from ..bom1.seconds_to_timestamp import seconds_to_timestamp
import pandas as pd
import numpy as np

def generate_path(df, filetype = '.wav'):
    '''
    Generate an outpath from a dataframe
    '''
    assert np.all(df.columns == ['ID', 't1', 't2', 'class']), 'Columns should be [ID, t1, t2, class]'
    
    t = df[['t1', 't2']].mean(axis=1).tolist()
    ts = [seconds_to_timestamp(x) for x in t]
    
    #Constructs the time e.g. H_00_M_01_S_05_SS_00. 
    ts = ['H_' + x.split(':')[0] + '_M_' + x.split(':')[1] + '_S_' + x.split(':')[2].split('.')[0] + '_SS_' + x.split(':')[2].split('.')[1] for x in ts]

    outpaths = 'ID_' + df['ID'] + \
              '_TS_' + ts + \
               '_C_' + df['class'].astype(str) + filetype
    
    return outpaths