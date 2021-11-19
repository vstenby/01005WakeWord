from ..bom1.seconds_to_timestamp import seconds_to_timestamp
import pandas as pd
import numpy as np

def generate_path(df, filetype = '.wav'):
    '''
    Generate an outpath from a dataframe
    '''
    assert np.all(df.columns == ['ID', 't1', 't2', 'class']), 'Columns should be [ID, t1, t2, class]'
    
    ts1 = [seconds_to_timestamp(x) for x in df['t1'].tolist()]
    ts2 = [seconds_to_timestamp(x) for x in df['t2'].tolist()]
    
    #Constructs the time e.g. H_00_M_01_S_05_SS_00 indicating t1, and then similar for t2.
    ts = ['H_' + x.split(':')[0] + '_M_' + x.split(':')[1] + '_S_' + x.split(':')[2].split('.')[0] + '_SS_' + x.split(':')[2].split('.')[1] + \
         '_H_' + y.split(':')[0] + '_M_' + y.split(':')[1] + '_S_' + y.split(':')[2].split('.')[0] + '_SS_' + y.split(':')[2].split('.')[1] for x, y in zip(ts1, ts2)]

    outpaths = 'ID_' + df['ID'] + \
              '_TS_' + ts + \
               '_C_' + df['class'].astype(str) + filetype
    
    return outpaths