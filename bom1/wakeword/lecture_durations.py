import pandas as pd

def lecture_durations(local_path = '/zhome/55/f/127565/Desktop/01005WakeWord/'):
    durations = pd.read_csv(local_path + 'csv/links.csv')[['ID', 'duration']]
    return dict(zip(durations['ID'].tolist(), durations['duration'].tolist()))