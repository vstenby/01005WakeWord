import numpy as np
import pandas as pd
import argparse
import tqdm
import os

import pandas as pd
import numpy as np
import subprocess

import bom1 as bom1

def main():
    
    data = pd.read_csv('./csv/data.csv')

    parser = argparse.ArgumentParser()
    
    #Add arguments
    parser.add_argument('--seed', default=42, type=int, help='random seed for numpy.')    
    parser.add_argument('--ar', default=22050, type=int, help='sample rate')
    parser.add_argument('--folder', default='./data/', type=str, help='folder name.')
    
    args = parser.parse_args()
    assert len([x for x in os.listdir(args.folder) if x.endswith('.wav')]) == 0, 'Export folder should be empty.'
    
    #Set the seed for numpy.
    np.random.seed(args.seed)
    
    for ID, grp in tqdm.tqdm(data.groupby('ID')):
        grp = grp.sort_values(by='t1')
        grp = grp[['ID', 't1', 't2']]

        for k, row in enumerate(grp.iterrows()):
            _, row = row
            outpath = f'{args.folder}{ID}_{str(k).zfill(2)}_1.wav'
            bom1.ffmpeg_clip(row['t1'], row['t2'], ID, outpath, ar=args.ar)
            
        duration = bom1.get_duration(ID)
        nclass0 = 0
        
        while nclass0 < len(grp):
            t = np.random.uniform(1, duration-1)
            t1 = t - 1
            t2 = t + 1
            no_overlap = np.all((t1 <= grp['t1'].to_numpy()) ==  (t2 <= grp['t1'].to_numpy()))
            if no_overlap:
                outpath = f'{args.folder}{ID}_{str(nclass0).zfill(2)}_0.wav'
                bom1.ffmpeg_clip(t1, t2, ID, outpath, ar=args.ar)
                nclass0 += 1
                
    return
    
if __name__ == '__main__':
    main()
