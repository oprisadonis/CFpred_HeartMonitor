import numpy as np
import pandas as pd
import neurokit2 as nk
import heartpy as hp
import math
def pick_features(all_measures, selected_measures):
    ppg_features = []
    for i in range(len(all_measures['bpm'])):
        row = []
        for j in selected_measures:
            value = all_measures[j][i]
            row.append(value)
        ppg_features.append(row)
    return ppg_features

def process_ppg(data, selected_features):
    working_data, measures = hp.process_segmentwise(data, sample_rate=100, segment_width=300, segment_overlap=0.5, segment_min_size=250,calc_freq=True)
    return pick_features(measures, selected_features)

def process_label(gamer, size,binary_sleepiness = True):
    data = []
    sleepiness = []
    with open('archive/gamer' + gamer + '-annotations.csv') as file:
        file.readline()
        for row in file:
            time, event, value = row.strip().split(',', 2)
            if event == 'Stanford Sleepiness Self-Assessment (1-7)':
                value = int(value)
                if binary_sleepiness == True:
                    if value < 4:
                        sleepiness.append(0)
                    else:
                        sleepiness.append(1)
                else:
                    sleepiness.append(value)
    sleepiness = sleepiness[2:]
    sleepiness = sleepiness[:-1]
    for i in range(len(sleepiness)):
        data = data + [sleepiness[i]] * math.ceil(size / len(sleepiness))

    return data[:size]

def normalize_data(df):
    for col in df.columns:
        if col != 'Sleepiness':
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def process_gamers(gamer,binary_sleepiness = True,features = ['bpm']):
    ppg_data1 = pd.read_csv("archive/gamer" + gamer + "-ppg-2000-01-01.csv")
    ppg_data2 = pd.read_csv("archive/gamer" + gamer + "-ppg-2000-01-02.csv")

    ppg_cleaned1 = nk.ppg_clean(ppg_data1["Red_Signal"], sampling_rate=100)
    ppg_cleaned2 = nk.ppg_clean(ppg_data2["Red_Signal"], sampling_rate=100)

    features_ppg1 = process_ppg(ppg_cleaned1, features)
    features_ppg2 = process_ppg(ppg_cleaned2, features)
    #print(features_ppg1)
    #print(features_ppg2)
    features_ppg = np.concatenate((features_ppg1, features_ppg2), axis=0,dtype=object)
    labels_ppg = process_label(gamer, len(features_ppg),binary_sleepiness)

    data_set = pd.DataFrame(features_ppg, columns=features)
    data_set['Sleepiness'] = labels_ppg
    return data_set

def tabel_generator(binary_sleepiness, normalize = False):
    features = ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2',
                'breathingrate', 'vlf', 'lf', 'hf', 'lf/hf', 'p_total','vlf_perc', 'lf_perc', 'hf_perc']#,'vlf_perc', 'lf_perc', 'hf_perc', 'lf_nu','hf_nu', 'segment_indices']
    d_set1 = process_gamers("1",binary_sleepiness = binary_sleepiness,features = features)
    d_set2 = process_gamers("2",binary_sleepiness = binary_sleepiness,features = features)
    d_set3 = process_gamers("3",binary_sleepiness = binary_sleepiness,features = features)
    d_set4 = process_gamers("4",binary_sleepiness = binary_sleepiness,features = features)
    d_set5 = process_gamers("5",binary_sleepiness = binary_sleepiness,features = features)
    col_names = features
    col_names.append('Sleepiness')
    if normalize == True:
        d_set1 = normalize_data(d_set1)
        d_set2 = normalize_data(d_set2)
        d_set3 = normalize_data(d_set3)
        d_set4 = normalize_data(d_set4)
        d_set5 = normalize_data(d_set5)

    d_set1.to_csv('paper_normalized/gamer1.csv')
    d_set2.to_csv('paper_normalized/gamer2.csv')
    d_set3.to_csv('paper_normalized/gamer3.csv')
    d_set4.to_csv('paper_normalized/gamer4.csv')
    d_set5.to_csv('paper_normalized/gamer5.csv')

    data_tabel = pd.concat([d_set1, d_set2, d_set3, d_set4, d_set5],names=col_names,ignore_index=True,axis=0)
    return data_tabel


# import os
# os.makedirs("data_table", exist_ok = True)
#
# data_binary_sleepiness = tabel_generator(binary_sleepiness=True)
# data_binary_sleepiness.to_csv("data_table\out_binary_sleepiness.csv",index=False)

data_non_binary_sleepiness = tabel_generator(binary_sleepiness=False, normalize=True)
# data_non_binary_sleepiness.to_csv("data_table\out_non_binary_sleepiness.csv",index=False)
#
# data_bs_normalized = tabel_generator(binary_sleepiness=True, normalize=True)
# data_bs_normalized.to_csv("data_table\out_bs_normalized.csv",index=False)
#
# data_non_bs_normalized = tabel_generator(binary_sleepiness=False, normalize=True)
# data_non_bs_normalized.to_csv("data_table\out_non_bs_normalized.csv",index=False)