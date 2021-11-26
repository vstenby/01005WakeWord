from re import split
import pandas as pd
import bom1.timestamp_to_seconds

def add_to_positive_class(path):
    links_csv = '../../csv/links.csv'
    data_csv = '../../csv/data.csv'

    links = pd.read_csv(links_csv)
    data = pd.read_csv(data_csv)

    split_path = path.split('_')
    lecture_id = '0_' + split_path[2]
    timestamp = split_path[5] + ':' + split_path[7] + ':' + split_path[9]
    seconds = bom1.timestamp_to_seconds(timestamp)

    lecture_row = links[links['ID'] == lecture_id]
    row_to_append = {'semester': lecture_row['semester'].item(), 
                     'skema': lecture_row['skema'].item(), 
                     'title': lecture_row['title'].item(), 
                     'ID': lecture_row['ID'].item(), 
                     't': seconds}

    df = pd.DataFrame(row_to_append, index=[0])

    data = data.append(df, ignore_index=True)
    data.to_csv(data_csv, index=False)
    

if __name__ == '__main__':
    path = 'ID_0_zxt50648_TS_H_00_M_37_S_29_SS_27_H_00_M_37_S_39_SS_27_C_0.wav'
    add_to_positive_class(path)