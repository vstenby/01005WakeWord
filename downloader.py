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

    #Append the train, test and validation columns to the data.
    train, test, val = bom1.get_splits()
    data['train'] = data['ID'].isin(train)
    data['test']  = data['ID'].isin(test)
    data['val']   = data['ID'].isin(val)
    
    parser = argparse.ArgumentParser()
    
    #Add arguments
    parser.add_argument('--seed', default=42, type=int, help='random seed for numpy.')    
    parser.add_argument('--ar', default=22050, type=int, help='sample rate')
    parser.add_argument('--exportpath', default='export', type=str, help='folder name.')
    parser.add_argument('--splits', default="train, test, val", type=str, help='which splits should be exported')
    parser.add_argument('--ratio', default=1, type=int, help='class ratio. nclass0 = round(nclass1 * ratio)')
    
    args = parser.parse_args()
    assert ~os.path.exists(args.exportpath), 'outfolder should not exist.'
    os.mkdir(args.exportpath)
    
    #Set the seed for numpy.
    np.random.seed(args.seed)
    
    #Fetch what splits we should get.
    splits = args.splits.replace(' ','')
    splits = splits.lower()
    splits = splits.split(',')
        
    assert np.all(np.array([x in ['train', 'test', 'val'] for x in splits])), 'Invalid usage of --splits'

    for split in splits:
        os.mkdir(os.path.join(args.exportpath, split))
        
        #Get the current data split.
        data_split_class1 = data.loc[data[split]]
        data_split_class1 = data_split_class1[['ID', 't1', 't2']]
        
        #Append the negative cases.
        print(f'Constructing negative cases for {split}.')
        data_split = bom1.append_negative_cases(data_split_class1, method = 'random', ratio=1)
        print(f'Negative cases appended.')
        
        #Construct the outpath
        data_split['outpath'] = './' + args.exportpath + '/' + split + '/' + data_split['ID'] + '_t' + data_split[['t1', 't2']].mean(axis=1).astype(str) + '_c' + data_split['class'].astype(str) + '.wav'
        
        for _, (ID, t1, t2, _, outpath) in tqdm.tqdm(data_split.iterrows(), total = data_split.shape[0], desc=split):
            bom1.ffmpeg_clip(t1, t2, ID, outpath, ar=args.ar)
        
        #Save the csv in the export folder as well.
        data_split.to_csv(f'./{args.exportpath}/{split}/{split}.csv', index=False)
        
        
    return
    
if __name__ == '__main__':
    main()
