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
    parser.add_argument('--seed', default=42, type=int, help='random seed for numpy.')    
    parser.add_argument('--ar', default=22050, type=int, help='sample rate')
    parser.add_argument('--folder-name', default='export', type=str, help='folder name.')
    parser.add_argument('--splits', default="train, val, test", type=str, help='which splits should be exported')
    parser.add_argument('--balance', default='1:1', type=str, help='define the balance')
    parser.add_argument('--threads', default=19, type=int, help='how many threads we should download on')
    parser.add_argument('--normalize', default=False, action='store_true', help='whether or not downloaded clips should be normalized.')
    #Add more arguments.
    
    args = parser.parse_args()
    assert ~os.path.exists(args.folder_name), 'outfolder should not exist.'
    os.mkdir(args.folder_name)
    
    #Set the seed for numpy.
    np.random.seed(args.seed)
    
    #Fetch what splits we should get.
    splits = args.splits.replace(' ','')
    splits = splits.lower()
    splits = splits.split(',')
        
    assert np.all(np.array([x in ['train', 'test', 'val'] for x in splits])), 'Invalid usage of --splits'

    train, val, test = wf.get_balanced_splits(balance=args.balance, seed=args.seed, splits = splits)
    
    for split in splits:
        n0, n1 = eval(split).groupby(['class']).size()
        print(f'{split.ljust(5)} has {n1} positives and {n0} negatives, or a ratio of 1:{int(np.round(n0/n1))}.')
    
    prompt = input(f'Do you want to download? [y/n] ').lower().strip()
    
    if (prompt != 'y') and (prompt != ''):
        return
    
    
    for split in splits:
        os.mkdir(os.path.join(args.folder_name, split))
        split_df = eval(split)
        
        t1 = split_df['t1'].tolist()
        t2 = split_df['t2'].tolist()
        ID = split_df['ID'].tolist()
        outpath = wf.generate_path(split_df).tolist()
        
        #Append all of the other funky stuff to the outpath.
        outpath = [os.path.join(args.folder_name, split, x) for x in outpath]
        
        bom1.clip(t1, t2, ID, outpath, desc = f'Downloading {split}', ar = args.ar, threads=args.threads, normalize = args.normalize)
        
    return
    
if __name__ == '__main__':
    main()
