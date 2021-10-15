import numpy  as np
import pandas as pd
import subprocess
import os 

# --- Functions related to clipping and video.dtu.dk ---
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

def clip(t1, t2, ID, pathout, ar=44100):
    '''
    Clip from a lecture.
    '''
    t1_timestamp = seconds_to_timestamp(t1) 
    t2_timestamp = seconds_to_timestamp(t2) 
    duration     = seconds_to_timestamp(t2-t1) 
    url          = stream_link(ID)
    bashcmd = f'ffmpeg -ss {t1_timestamp} -i "{url}" -t {duration} -q:a 0 -map a {pathout} -loglevel error'
    rtrn = subprocess.call(bashcmd, shell=True)    
    assert rtrn == 0, 'clip failed.'  
    return 

def download(ID, pathout):
    '''
    Download a full lecture.
    '''
    url = stream_link(ID)
    bashcmd = f'ffmpeg -i "{url}" -q:a 0 -map a {pathout} -loglevel error'
    rtrn = subprocess.call(bashcmd, shell=True)
    assert rtrn == 0, 'download failed.'
    return 


# --- End of functions related to video.dtu.dk ---

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

def get_mask(ID, t, n, label = False):
    '''
    Gets a mask where the keywords are. This is similar to get_targets.
    '''
    
    assert len(t.shape) == 2, 't should have [n x 2] shape.'
    assert t.shape[-1]  == 2, 't should have [n x 2] shape.'
    
    #First, we want to get the duration.
    duration   = get_duration(ID)
    
    t_linspace = np.linspace(0, duration, n)
    targets    = np.zeros_like(t_linspace)
    
    t1s = t[:,0]
    t2s = t[:,1]
    
    mask_value = 1
    for t1, t2 in zip(t1s, t2s):
        targets[(t1 <= t_linspace)&(t_linspace <= t2)] = mask_value
        if label: mask_value += 1
        
    return t_linspace, targets

def get_targets(ID, t, n, delay = 0, target_duration = 1):
    '''
    Returns the targets of a lecture. 
    
    Inputs:
        ID              : ID to the lecture
        t               : array of size n x 2, where t[:,0] is t1 (start of wakeword) and t[:,1] is t2 (end of wakeword)
        n               : desired output length
        delay           : delay from t2 until target in seconds.
        target_duration : how long target should be 1 after t2+delay.
    '''
    
    assert len(t.shape) == 2, 't should have [n x 2] shape.'
    assert t.shape[-1]  == 2, 't should have [n x 2] shape.'
    
    #First, we want to get the duration.
    duration   = get_duration(ID)
    
    t_linspace = np.linspace(0, duration, n)
    targets    = np.zeros_like(t_linspace)
    
    #Fetch end of keyword utterances and add delay.
    t2s = t[:,1] + delay
    
    for t2 in t2s:
        timediffs = t_linspace - t2
        targets[(timediffs > 0)&(timediffs <= target_duration)] = 1
    
    return t_linspace, targets

def get_random_clip(duration):
    '''
    Get a random clip with 0 <= t1 and t2 <= duration.
    '''
    t  = np.random.uniform(1, duration-1)
    t1 = t - 1
    t2 = t + 1
    return (t1, t2)
 
def overlapping(timestamps1, timestamps2):    
    t_x = np.mean(timestamps1)
    t_y = np.mean(timestamps2)
    return np.abs(t_x - t_y) < 2 #since clips are 2 seconds long.
 
def append_negative_cases(dataframe_class1, method='random', ratio=1):
    '''
    Takes a dataframe with columns [ID, t1, t2] with positive cases and appends negative cases.
    '''
    if method == 'random':
        assert type(ratio) is int, 'ratio should be an integer.'
        
        n_class1 = dataframe_class1.groupby('ID').size().reset_index(name='n')
        n_class0 = n_class1

        ratio = 1

        n_class0['n'] = np.round(n_class0['n'] * ratio)
        n_class0['IDduration'] = [get_duration(x) for x in n_class0['ID']]

        dataframe_class0 = pd.DataFrame(columns = dataframe_class1.columns)

        for _, (ID, n, duration) in n_class0.iterrows():
            n_exported = 0
            subset = dataframe_class1.loc[dataframe_class1['ID'] == 'ID']
            while n_exported < n:
                timestamp = get_random_clip(duration)
                if not subset[['t1', 't2']].apply(lambda x : overlapping(timestamp, x), axis=1).any():
                    dataframe_class0 = dataframe_class0.append(pd.DataFrame({'ID' : ID, 't1' : timestamp[0], 't2' : timestamp[1]}, index=[0]), ignore_index=True)
                    n_exported += 1

        assert n_class0[['ID', 'n']].equals(dataframe_class0.groupby('ID').size().reset_index(name='n')), 'Wrong number of clips exported.'
        
        dataframe_class0['class'] = 0
        dataframe_class1['class'] = 1
        dataframe = pd.concat([dataframe_class1, dataframe_class0])
    else:
        raise NotImplementedError(f'Method {method} is not implemented yet.')
        
    return dataframe