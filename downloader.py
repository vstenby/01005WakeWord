import numpy as np
import pandas as pd
import argparse
import tqdm
import os

import pandas as pd
import numpy as np
import subprocess

import bom1.wakeword as wf
import bom1.bom1 as bom1

def main():
        
    parser = argparse.ArgumentParser()
    
    #Add arguments
    parser.add_argument('--export-folder', default='export', type=str, help='name of the exported folder') 
    parser.add_argument('--seed', default=42, type=int, help='random seed for numpy.')    
    parser.add_argument('--ar', default=22050, type=int, help='sample rate')
    parser.add_argument('--splits', default="train, val, test", type=str, help='which splits should be exported')
    parser.add_argument('--balance', default='1:1', type=str, help='define the balance')
    parser.add_argument('--threads', default=19, type=int, help='how many threads we should download on')
    parser.add_argument('--normalize', default=False, action='store_true', help='whether or not downloaded clips should be normalized.')
    parser.add_argument('--cliplength', default=2.0, type=float, help='length of the clips')
    parser.add_argument('--data-folder', default='/work3/s164419/01005WakeWordData', help='point to data folder')
    #Add more arguments.
    
    args = parser.parse_args()
    assert ~os.path.exists(os.path.join(args.data_folder, args.export_folder)), 'export folder should not exist.'
    os.mkdir(os.path.join(args.data_folder, args.export_folder))
    
    #Set the seed for numpy.
    np.random.seed(args.seed)
    
    #Fetch what splits we should get.
    splits = args.splits.replace(' ','')
    splits = splits.lower()
    splits = splits.split(',')
        
    assert np.all(np.array([x in ['train', 'test', 'val'] for x in splits])), 'Invalid usage of --splits'

    train, val, test = wf.get_balanced_splits(balance=args.balance, seed=args.seed, splits = splits, cliplength=args.cliplength)
    
    for split in splits:
        if split == 'test':
            #Avoid spoilers!
            continue 
        n0, n1 = eval(split).groupby(['class']).size()
        print(f'{split.ljust(5)} has {n1} positives and {n0} negatives, or a ratio of 1:{int(np.round(n0/n1))}.')
    
    prompt = input(f'Do you want to download? [y/n] ').lower().strip()
    
    if (prompt != 'y') and (prompt != ''):
        return
    
    
    for split in splits:
        os.mkdir(os.path.join(args.data_folder, args.export_folder, split))
        split_df = eval(split)
        
        t1  = split_df['t1'].tolist()
        t2  = split_df['t2'].tolist()
        
        prepath = os.path.join(args.data_folder, 'lectures')
        url = (prepath + '/' + split_df['ID'] + '.wav').tolist() 
        
        #Construct the outpath.
        prepath = os.path.join(args.data_folder, args.export_folder, split)
        outpath = (prepath + '/' + wf.generate_path(split_df)).tolist()
        
        bom1.clip(t1, t2, url, outpath, desc = f'Downloading {split}', ar = args.ar, threads=args.threads, normalize = args.normalize)
        
    return
    
if __name__ == '__main__':
    main()
