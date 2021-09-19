import numpy  as np
import pandas as pd
import subprocess
import os 

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

def ffmpeg_clip(t1, t2, ID, pathout, ar=44100):
    '''
    Export a clip.
    Input:
        t1:  start time (seconds)
        t2:  end time (seconds)
        url: video.dtu.dk link
    '''
    assert 0 <= t1, f'Invalid value of t1: {t1}'
    assert pathout.endswith('.wav'), 'only supports .wav files'
    
    #Replace letters causing trouble.
    pathout = pathout.replace(' ','_')\
                     .replace(',','')\
                     .replace("'","") 
    
    t1_timestamp = seconds_to_timestamp(t1) 
    t2_timestamp = seconds_to_timestamp(t2) 
    duration = seconds_to_timestamp(t2-t1) 
    
    #Get the stream url from the ID.
    url = stream_link(ID)
    
    bashcmd = f'ffmpeg -ss {t1_timestamp} -i "{url}" -t {duration} -ar {ar} -map a {pathout} -loglevel error'    
    
    rtrn = subprocess.call(bashcmd, shell=True)

    return rtrn