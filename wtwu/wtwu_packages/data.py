import scipy.io
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
import os
from scipy.signal import welch



file_path = '/home/mariorocha/code/SeanDominique/will-they-wake-up/data/raw/physionet.org/files/i-care/2.1/training/0284/'
##TODO Change the path using os, if possible making the patient a variable.
def get_eeg_paths(directory):
    '''
    This function gets all the EEG file paths from a directory input, sorted over time.
    '''

    eeg_files = []

    for root, _, files in os.walk(directory):

        for file in files:

            if file.endswith("EEG.mat"):

                absolute_path = os.path.abspath(os.path.join(root, file))
                eeg_files.append(absolute_path)

    return sorted(eeg_files)


def recover_eegs_and_hours(patient,scaler='Standard'):
    '''
    This function gets extracts all EEGs from a list of file paths (patient input) and puts it in a list.
    It also gives a list of all the 'Hours after cardiac arrest' per file on a separate list.
    It would be better if we could just put the patient number - TODO


    It takes as scaler the following: 'Standard', 'MinMax' or 'Robust'.
    The default value is 'Standard'.

    If you think the last files of the directory are corrupted (or not completely downloaded yet)
    use the argument 'stop_before' = number of files to skip in the end.

    There's a commented line if we want to do the mean of all channels.

    '''

    EEG_file_list = get_eeg_paths(patient)
    hours =[]

    if scaler == 'MinMax':
        scaler = MinMaxScaler()
    if scaler == 'Standard':
        scaler = StandardScaler()
    if scaler == 'Robust':
        scaler = RobustScaler()

    EEG_list = []
    for file_path in EEG_file_list[1:-1]:

        eeg = scipy.io.loadmat(file_path)

        eeg = eeg['val']
        eeg = eeg.astype(float)

        #    temp_line = line.reshape(1, -1)
        #    temp_line = scaler.fit_transform(temp_line)
        #    line = temp_line.reshape(-1)
        # eeg = np.mean(eeg,axis=0)
        if eeg.shape == (20, 7372800):
            EEG_list.append(eeg)
            hour = float(file_path[-11:-8])
            print(hour)
            hours.append(hour)



    EEG_list = np.array(EEG_list)
    hours = np.array(hours)
    hours = hours.reshape(1,-1)

    return EEG_list,hours

def reduce_EEGs(list_of_EEGs, rate_of_reduction = 5, original_freq = 500):
    '''
    This function takes a list of EEG spectra, the rate_of_reduction of the frequency
    and the original_frequency, the last two defaulted at 5 and 500.

    That means that an EEG with an original frequency of 500 Hz if passed through
    with rate_of_reduction = 5, will return an EEG of frequency 500/5, that is, 100 Hz.

    The rate_of_reduction variable will be rounded before any calculations to avoid problems
    trying to create arrays of non-integer sizes.

    '''


    rate_of_reduction = np.round(rate_of_reduction)


    def single_reduction(EEG):
            if len(EEG) % rate_of_reduction == 0:

                print(f'EEG reduced to a {original_freq/rate_of_reduction} Hz frequence.')
                return EEG.reshape(-1, int(rate_of_reduction) ).mean(axis=1)

            else:
                print('Data will be cut due to undivisible length.')

                remaining = int(len(EEG) % rate_of_reduction)
                print(f'Points lost: {remaining}')

                EEG = EEG[:-remaining]
                print(f'EEG reduced to a {original_freq/rate_of_reduction} Hz frequence.')

                return EEG.reshape(-1, int(rate_of_reduction) ).mean(axis=1)
    new_list_of_EEGs = []
    for EEG in list_of_EEGs:
        EEG = single_reduction(EEG)
        new_list_of_EEGs.append(EEG)
    return new_list_of_EEGs

def sampling_EEGs(list_of_EEGs, fs=100, sampling_rate=600, sampling_size=15,hours=None):
    '''
    This function takes a list of EEGs, their frequency, the sampling rate in seconds
    (every 10 min = 600), the sampling size in seconds, and if available, the list of
    hours after cardiac arrest.

    It returns splits of the EEGs every (sampling_rate) seconds of length (sampling_size)
    seconds. If a list of hours was given, it also returns a list of decimal times in hours.
    This list can later be transformed in HH:MM easily, if needed.

    '''


    splits = []
    split_time = []
    i_EEG = 0

    for EEG in list_of_EEGs:

        i = 0
        while ((sampling_rate*i) + sampling_size)*fs < len(EEG):

            splits.append(EEG[(sampling_rate*i)*fs:((sampling_rate*i)+sampling_size)*fs])

            if hours != None:

                split_time.append(float(hours[i_EEG])+((fs/sampling_rate)*i))

            i += 1

        i_EEG += 1

    splits = np.array(splits)
    if len(split_time)>0:
        split_time = np.array(split_time)
    return splits, split_time





def get_psds(EEG_list,fs=100, mode='channels', hours=None,input_type='list'):
    '''
    This function takes an array of EEGs and returns the PSDs.
    It can have two modes: 'channels' or 'time':

    On 'channels' mode, it will return a PSD for each channel
    if given a list of EEGs of a single patient from a single raw file.

    On 'time' mode, it will take as input a list containing EEGs of a single
    channel or an average of a single patient and return a dataframe containing
    all PSDs as columns and their 'hours after cardiac arrest' as index.

    input_type = ['list' ,'array']


    '''

    if hours != None:


        hours_df = pd.DataFrame(hours,columns='time')

    if input_type == 'list':
        psds = []
        for eeg in EEG_list:
            f, temp_psd = welch(eeg, fs=fs, nperseg=1024)
            psds.append(temp_psd)
        psds_df = pd.DataFrame(psds)
        if mode == 'time' and hours.shape[1] == psds_ar.shape[1]:
            psds_df = pd.concat([psds_df, hours_df], axis=1)
            psds_df = psds_df.groupby(by='hours',as_index=True).mean()
            return psds_df, f
        elif mode == 'time':
            return psds_df, f
        elif mode == 'channels':
            return psds, f
        else:
            print('get_psds: unrecognised mode.')
            return None

    if input_type == 'array':
        psds_ar = np.zeros([EEG_list.shape[1],np.ceil(fs/2)])
        for i in range(EEG_list.shape[1]):
            f, psds_ar[i,:] = welch(psds[i,:], fs=fs, nperseg=1024)
        psds_df = pd.DataFrame(psds_ar)

        if mode == 'time' and hours.shape[1] == psds_ar.shape[1]:
            psds_df = pd.concat([psds_df, hours_df], axis=1)
            psds_df = psds_df.groupby(by='hours',as_index=True).mean()
            return psds_df, f
        elif mode == 'time':
            return psds_df, f
        elif mode == 'channels':
            return psds_ar, f
        else:
            print('get_psds: unrecognised mode.')
            return None

    else:
        print('get_psds: bad input type')
        return None
