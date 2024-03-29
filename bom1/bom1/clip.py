from .timestamp_to_seconds import *
from .seconds_to_timestamp import *
from .fix_outpath import *
from .parser import *
from .fetch_ID import *
from .stream_link import *
import subprocess

import multiprocessing
from tqdm import tqdm

def ffmpeg_clip(t1, duration, url, outpath, args):
    '''
    Export a clip.
    Input:
        t1:        start time (timestamp)
        duration:  duration   (timestamp)
        url:       stream link
    '''
    
    if outpath.endswith('.mp3') or outpath.endswith('.wav'):
        if duration is not None:
            bashcmd = f'ffmpeg -y -ss {t1} -i "{url}" -t {duration} -ar {args.ar} -q:a 0 -map a {outpath} -loglevel error'
        else:
            #If duration is None, then we should just download the entire thing.
            bashcmd = f'ffmpeg -y -ss {t1} -i "{url}" -ar {args.ar} -q:a 0 -map a {outpath} -loglevel error'
    elif outpath.endswith('.mp4') or outpath.endswith('.gif'):
        bashcmd = f'ffmpeg -y -ss {t1} -i "{url}" -t {duration} {outpath} -ar {args.ar} -loglevel error'
    else:
        raise ValueError('Wrong output format.')

    rtrn = subprocess.call(bashcmd, shell=True)
    
    assert rtrn == 0, 'Download failed.'
        
    if args.normalize and outpath.endswith('.mp4'):
        #This might fail if you don't have ffmpeg-normalize installed. pip3 install ffmpeg-normalize. Normalization also only seems to work with mp4.
        normalize_rtrn = subprocess.call(f'ffmpeg-normalize {outpath} -o {outpath} -c:a aac -b:a 192k -f', shell=True, 
                                         stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        assert normalize_rtrn == 0, 'Normalize failed.'
        
    return rtrn

def clip(t1, t2, url, outpath, args = parser().parse_args([]),
        #Here, we set all arguments that can be used by ffmpeg_clip. 
        normalize = None,
        silent    = None,
        threads   = None,
        ar        = None,
        desc      = None,
        ):

    if normalize is not None: args.normalize = normalize
    if silent is not None:    args.silent = silent
    if threads is not None:   args.threads = threads
    if ar is not None:        args.ar = ar
    if desc is not None:      args.desc = desc
    
    #Make sure that all arguments are good to go.
    outpath  = [fix_outpath(x) for x in outpath]
    
    #This can be written in a more clever way, but for now it'll work.
    url      = [stream_link(fetch_ID(x)) if 'video.dtu.dk' in x else stream_link(x) if not x.endswith('.wav') else x for x in url] #Returns stream_link if url or ID and doesn't end with .wav, otherwise just returns x.
    t1       = [seconds_to_timestamp(x) for x in t1]   #make sure all t1 are timestamps.
    t2       = [timestamp_to_seconds(x) for x in t2]   #make sure all t2 are seconds. 
    duration = [seconds_to_timestamp(y - timestamp_to_seconds(x)) for x, y in zip(t1, t2)] #This will fail if t1 and t2 have different lengths.

    pool = multiprocessing.Pool(processes=args.threads)
    
    if not args.silent:
        pbar = tqdm(total = len(t1), desc=args.desc)
        
        def update(*a):
            pbar.update()
        
        callback = update
    else:
        callback = None

    for i in range(len(t1)):
        pool.apply_async(ffmpeg_clip, args=(t1[i], duration[i], url[i], outpath[i], args), callback=callback)
        
    pool.close()
    pool.join()
        
    return 

def ffmpeg_download(url, outpath, args):
    '''
    Download an entire lecture in parallel.
    '''
    
    if outpath.endswith('.mp3') or outpath.endswith('.wav'):
        bashcmd = f'ffmpeg -y -i "{url}" -q:a 0 -map a {outpath} -loglevel error'
    elif outpath.endswith('.mp4') or outpath.endswith('.gif'):
        bashcmd = f'ffmpeg -y -i "{url}" {outpath} -loglevel error'
    else:
        raise ValueError('Wrong output format.')
            
    rtrn = subprocess.call(bashcmd, shell=True)
    
    assert rtrn == 0, 'Download failed.'
                
    return rtrn

def download(url, outpath, args = parser().parse_args([]),
             #Here, we set all arguments that can be used by ffmpeg_download. 
             silent    = None,
             threads   = None,
             desc      = None,
             ):
    
    if silent is not None:    args.silent = silent
    if threads is not None:   args.threads = threads
    if desc is not None:      args.desc = desc
    
    #Make sure that all arguments are good to go.
    outpath  = [fix_outpath(x) for x in outpath]
    
    #This can be written in a more clever way, but for now it'll work.
    url      = [stream_link(fetch_ID(x)) if 'video.dtu.dk' in x else stream_link(x) for x in url] #The url is either video.dtu.dk link or an ID to a video.
    
    pool     = multiprocessing.Pool(processes=args.threads)
    
    if not args.silent:
        pbar = tqdm(total = len(url), desc=args.desc)
        
        def update(*a):
            pbar.update()
        
        callback = update
    else:
        callback = None

    for i in range(len(url)):
        pool.apply_async(ffmpeg_download, args=(url[i], outpath[i], args), callback=callback)
        
    pool.close()
    pool.join()
        
    return 