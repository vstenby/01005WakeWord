import os
from ..bom1.timestamp_to_seconds import timestamp_to_seconds

def info_from_path(path):
    #Extract the information from the path.
    basename = os.path.basename(path)
    extension = '.' + basename.split('.')[-1]
    ID = '_'.join(basename.split('_')[1:3])
    ts1 = basename.split('_')[5] + ':' + basename.split('_')[7] + ':' + basename.split('_')[9] + '.' + basename.split('_')[11]
    ts2 = basename.split('_')[13] + ':' + basename.split('_')[15] + ':' + basename.split('_')[17] + '.' + basename.split('_')[19]
    
    t1 = timestamp_to_seconds(ts1)
    t2 = timestamp_to_seconds(ts2)

    target = int(basename.split('_')[-1].split('.')[0])
    
    return ID, t1, t2, target
    
    