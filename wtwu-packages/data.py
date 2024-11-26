import scipy.io
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
import os
from scipy.signal import welch



file_path = '/home/mariorocha/code/SeanDominique/will-they-wake-up/data/raw/physionet.org/files/i-care/2.1/training/0284/'
##TODO Change the path using os, if possible making the patient a variable.
def get_eeg_paths(directory):
    eeg_files = []

    for root, _, files in os.walk(directory):

        for file in files:

            if file.endswith("EEG.mat"):

                absolute_path = os.path.abspath(os.path.join(root, file))
                eeg_files.append(absolute_path)

    return sorted(eeg_files)


def recover_eegs_and_hours(patient,scaler='Standard',stop_before=0):


    EEG_file_list = get_eeg_paths(patient)
    hours =[]
    for file in EEG_file_list:
        hour = file[-11:-8]
        hours.append(hour)
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
    if scaler == 'Standard':
        scaler = StandardScaler()
    if scaler == 'Robust':
        scaler = RobustScaler()

    EEG_list = []
    for file_path in EEG_file_list[:-stop_before]:
        eeg = scipy.io.loadmat(file_path)
        eeg = eeg['val']
        eeg = eeg.astype(float)
        eeg = scaler.fit_transform(eeg)
        eeg = np.mean(eeg,axis=0)
        EEG_list.append(eeg)
    hours = pd.DataFrame(hours,columns=['hours'])

    return EEG_list,hours

def get_psds(EEG_list,hours):
    psds = []
    for eeg in EEG_list:
        f, temp_psd = welch(eeg, fs=500, nperseg=1024)
        psds.append(temp_psd)
    psds_df = pd.DataFrame(psds)
    psds_df = pd.concat([psds_df, hours], axis=1)
    psds_df = psds_df.groupby(by='hours',as_index=True).mean()
    return psds_df
