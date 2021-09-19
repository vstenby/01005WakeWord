import numpy as np
import pandas as pd
import argparse
import tqdm

import pandas as pd
import numpy as np
import subprocess

def timestamp_to_seconds(ts):
    '''
    Converts a timestamp to seconds.
    '''
    assert ts.count(':') <= 2, f"Too many :'s in the timestamp: {ts}"
    assert ts.count('.') <= 1, f"Too many .'s in the timestamp: {ts}"
    sss_ = 0
    
    if ts.count('.') == 1:
        ts, sss = ts.split('.')
        sss_ = 0
        for i, n in enumerate(sss):
            sss_ += 1/(10**(i+1)) * float(n)
            
    seconds = float(sum(int(x) * (60**i) for i, x in enumerate(reversed(ts.split(':'))))) + sss_
    
    assert 0 <= seconds, f"Something went wrong with loading the timestamp. {ts}"
    
    return seconds

def seconds_to_timestamp(seconds):
    '''
    Convert the seconds to HH:MM:SS:SSS timestamp.
    '''
    sss = np.round(seconds - np.floor(seconds),2)

    m, s = divmod(np.floor(seconds), 60)
    h, m = divmod(m, 60)

    sss = str(sss).split('.')[-1].ljust(2,'0')

    h = str(int(h)).zfill(2); m = str(int(m)).zfill(2); s = str(int(s)).zfill(2);

    return ':'.join([h,m,s]) + '.' + sss
    
def stream_link(ID):
    '''
    Fetches a stream link from ID. 
    '''
    return f'https://dchsou11xk84p.cloudfront.net/p/201/sp/20100/playManifest/entryId/{ID}/format/url/protocol/https'
    
def download_link(ID):
    '''
    Returns a download link from ID.
    '''
    return f'https://dchsou11xk84p.cloudfront.net/p/201/sp/20100/playManifest/entryId/{ID}/format/download/protocol/https/flavorParamIds/0'

def get_duration(ID):
    '''
    Returns the duration of a video.
    '''
    import cv2 as cv
    
    #Read the video capture from cv2.
    cap = cv.VideoCapture(stream_link(ID))
    return int(cap.get(cv.CAP_PROP_FRAME_COUNT))/cap.get(cv.CAP_PROP_FPS)

def fetch_ID(url):
    '''
    Takes a video.dtu.dk link and returns the video ID.
    
    TODO: This should make some assertions about the url.
    '''
    return '0_' + url.split('0_')[-1].split('/')[0]

def ffmpeg_clip(t1, t2, ID, pathout, normalize=False):
    '''
    Export a clip.
    Input:
        t1:  start time (seconds)
        t2:  end time (seconds)
        url: video.dtu.dk link
    '''
    assert 0 <= t1, f'Invalid value of t1: {t1}'
        
    #Replace letters causing trouble.
    pathout = pathout.replace(' ','_')\
                     .replace(',','')\
                     .replace("'","") 
    
    t1_timestamp = seconds_to_timestamp(t1) 
    t2_timestamp = seconds_to_timestamp(t2) 
    duration = seconds_to_timestamp(t2-t1) 
    
    #Get the stream url from the video.dtu.dk url.
    url = stream_link(ID)
    
    if pathout.endswith('.mp3') or pathout.endswith('.wav'):
        bashcmd = f'ffmpeg -ss {t1_timestamp} -i "{url}" -t {duration} -q:a 0 -map a {pathout} -loglevel error'
    elif pathout.endswith('.mp4') or pathout.endswith('.gif'):
        bashcmd = f'ffmpeg -ss {t1_timestamp} -i "{url}" -t {duration} {pathout} -loglevel error'
    else:
        raise ValueError('Wrong output format.')
            
    rtrn = subprocess.call(bashcmd, shell=True)
    
    if normalize and pathout.endswith('.mp4'):
        #This might fail if you don't have ffmpeg-normalize installed. pip3 install ffmpeg-normalize. Normalization also only seems to work with mp4.
        normalize_rtrn = subprocess.call(f'ffmpeg-normalize {pathout} -o {pathout} -c:a aac -b:a 192k -f', shell=True, 
                                         stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
    return rtrn






def main():
    
    data = pd.read_csv('./csv/data.csv')

    parser = argparse.ArgumentParser()
    
    #Add arguments
    parser.add_argument('--n', default=len(data), type=int, help='how many negative classes should be downloaded. default is balanced.')
    
    args = parser.parse_args()

    for ID, grp in tqdm.tqdm(data.groupby('ID')):
        grp = grp.sort_values(by='t1')
        grp = grp[['ID', 't1', 't2']]

        for k, row in enumerate(grp.iterrows()):
            _, row = row
            outpath = f'./data/{ID}_{str(k).zfill(2)}_1.wav'
            ffmpeg_clip(row['t1'], row['t2'], ID, outpath)

        duration = get_duration(ID)
        nclass0 = 0

        while nclass0 < len(grp):
            t = np.random.uniform(1, duration-1)
            t1 = t - 1
            t2 = t + 1
            no_overlap = np.all((t1 <= grp['t1'].to_numpy()) ==  (t2 <= grp['t1'].to_numpy()))
            if no_overlap:
                outpath = f'./data/{ID}_{str(nclass0).zfill(2)}_0.wav'
                ffmpeg_clip(t1, t2, ID, outpath)
                nclass0 += 1
            
        break
            
    return
    
if __name__ == '__main__':
    main()
