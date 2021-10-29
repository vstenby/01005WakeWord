import pandas as pd

def lecture_durations():
    durations = pd.read_csv('./csv/links.csv')[['ID', 'duration']]
    return dict(zip(durations['ID'].tolist(), durations['duration'].tolist()))