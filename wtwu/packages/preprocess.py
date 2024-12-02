import os

from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler

from wtwu.params import *
from wtwu.packages.storage import get_list_of_patients, import_data

import pandas as pd
import numpy as np
import scipy.io
from scipy.signal import welch, butter, filtfilt
from mne.filter import resample, notch_filter, filter_data

####################### PREPROCESSING #######################

def preprocess(patients=[]):
    """
    Prend une liste de patient (eg. ["0284", "0341", "1024"]). Preprocess la donnée EEG de chacun et upload sur GCS.
    Cette fonction suppose que la donnée des patients est en local ou déja sur GCS.
    """

    # BUCKET_NAME =
    # PATIENT_DATA_PATH -> "i-care-2.0.physionet.org/training/"
    # local_processed_path -> destination
    # preprocessed_path -> GCS path for processed data


    # get patient data from GCS
    if DATA_TARGET == "gcs":
        if len(patients) == 0:
            patients = get_list_of_patients()

        for patient in patients:
            print(f"Traitement du patient {patient}...")

            # Import des données
            survived, eeg_data_headers, all_eeg_data = import_data(patient)

            if eeg_data_headers != "Error" and len(eeg_data_headers) > 0:

                fs = eeg_data_headers[0]['fs']
                hours = np.array([header['recording_hour'] for header in eeg_data_headers]).astype(np.float16)

                # Réduction des données
                    # TODO: try undersampling of 125Hz and 128Hz (in research paper)
                undersampled_eeg_data = undersample_eegs(all_eeg_data, target_freq=100, original_freq=fs)

                # Bandpass filter (0.1-40Hz)
                # `filter_data`` default uses FIR method which is more computationally intensive, but more stable. Using this over butterworth filter, an IIR method, because we are more focused on temporal analysis where phase distortion could be a problem.
                bandpassed_eeg_data = filter_data(undersampled_eeg_data,
                                                  sfreq=fs,
                                                  l_freq=0.01,
                                                  h_freq=40)

                # Notch filter
                hospital_location = "US" # TODO: collect EEG recording location from eeg_header
                if hospital_location == "EU":
                    notched_eeg_data = notch_filter(bandpassed_eeg_data, fs, np.arange(50,251,50))
                elif hospital_location == "US":
                    notched_eeg_data = notch_filter(bandpassed_eeg_data, fs, np.arange(60,241,60))

                # Artefact removal

                # clean_data() (AIRhythm -> unified EEG + ECG Channels)

                # calcul des PSD

                # Epoching
                # # Fenêtrage et normalisation
                list_of_splits, list_of_times = wtdata.sample_all(reduced_eeg_data, hours=hours)
                std = np.std(list_of_splits, axis=0)
                mean = np.mean(list_of_splits, axis=0)
                std = np.where(std == 0, 1, std)
                list_of_splits = (list_of_splits - mean) / std
                print("print fin prepro")

                # Waveform segmentation using rolling window

                # standardize (z-score normalization)


                ### Preprocessing patient data DONE


                # upload to relevant GCS bucket
                    # # Création du dossier local pour le patient
                    # gcs_file_path =
                    # os.makedirs(patient_local_path, exist_ok=True)

                    # # Écriture des métadonnées
                    # with open(f'{patient_local_path}/y.txt', 'a+') as f:
                    #     f.write(f'survived:{survived}\n')


            else:
                print(f"Patient {patient} : Données introuvables ou incorrectes.")

        else:
            # TODO: select specific patients
            print(f"Selection spécificque de patient n'est pas encore possible.")

            # if patient_local_path:
            #     upload_preprocessed_to_gcp(patient_local_path, bucket_name, preprocessed_path, patient)


    elif "local":
        print("uploading patient data from local...")
        print("failed") # TODO: logic
        return

    return


####################### ECHANTILLONAGE #######################
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
        #for line in eeg:
        #    temp_line = line.reshape(1, -1)
        #    temp_line = scaler.fit_transform(temp_line)
        #    line = temp_line.reshape(-1)
        # eeg = np.mean(eeg,axis=0)
        EEG_list.append(eeg)
        EEG_list = np.array(EEG_list)
        hour = file_path[-11:-8]
        hours.append(hour)

    hours = np.array(hours)
    hours = np.reshape(1,-1)

    return EEG_list,hours


### ✅ Undersampling functions
def undersample_eegs(list_of_eeg, original_freq= 500, new_freq=100):
    '''
    Returns a np.array of all a patient's EEG data with only relevant channels.

    This function takes a list of EEG spectra, the rate_of_reduction of the frequency
    and the original_frequency, the last two defaulted at 5 and 500.

    That means that an EEG with an original frequency of 500 Hz if passed through
    with rate_of_reduction = 5, will return an EEG of frequency 500/5, that is, 100 Hz.

    The rate_of_reduction variable will be rounded before any calculations to avoid problems
    trying to create arrays of non-integer sizes.

    '''
    return np.array([resample_eeg_data(eeg, original_freq, new_freq) for eeg in list_of_eeg])

def resample_eeg_data(raw_eeg, original_freq, new_freq, max_seconds=15):
    """
    Resamples raw_eeg_data, array-like of EEG spectra, to compress the data.

    Parameters:
        raw_eeg (np.ndarray):   TS data for a given EEG channel in the form on an 1D np.array.
        original_freq (float):  Original frequency in Hz
        new_freq (float):       Target frequency in Hz
    Returns:
        undersampled_eeg_data (np.ndarray): Resampled TS
    """

    resampling_ratio = original_freq / new_freq

    undersampled_eeg_data = resample(raw_eeg, down=resampling_ratio) # uses anti-aliasing

    # in case resampled array is too large
    max_samples = max_seconds * new_freq
    if len(undersampled_eeg_data) > max_samples:
        remaining = int(len(undersampled_eeg_data) % resampling_ratio)
        undersampled_eeg_data = undersampled_eeg_data[:max_samples]
        print(f'Data points lost: {remaining}')

    print(f'EEG reduced to a {original_freq/resampling_ratio} Hz frequence.')
    # EEG.reshape(-1, int(resampling_ratio) ).mean(axis=1) --> TODO: Why?

    return undersampled_eeg_data

### ✅ Epoching
def sampling_EEGs(list_of_EEGs, fs=100, sampling_rate=600, sampling_size=15,hours=None):
    '''
    This function takes a list of EEGs, their frequency, the sampling rate in seconds
    (every 10 min = 600), the sampling size in seconds, and if available, the list of
    hours after cardiac arrest.

    It returns splits of the EEGs every (sampling_rate) seconds of length (sampling_size)
    seconds. If a list of hours was given, it also returns a list of decimal times in hours.
    This list can later be transformed in HH:MM easily, if needed.

    Parameters:
    - list_of_EEGs (np.array): array of all a patient's EEG data
    - fs (int):                sampling frequency of EEG data (after undersampling in `undersample_eegs`)
    - sampling_rate (int):
    - sampling_size (int):     number of seconds the epoch
    - hours :

    Returns:


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


# TODO: Add data.sample_all()

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


####################### CLEANING THE DATA #######################
def clean_data():
    # TODO: make sure column names are

    # make sure everything is in the right shape

    # deal with different frequencies based on hospitals

    # get the right channels

    pass


def standardize(eeg_data: np.array, scaler="standard") -> np.array:
    """
    Return a numpy array with the standardized EEG data based on the given scaler
    """

    match scaler:
        case "standard":
            scale = StandardScaler()
        case "minmax":
            scale = MinMaxScaler()
        case "robust":
            scale = RobustScaler()
        case None:
            print("no standardization applied")
            return eeg_data

    eeg_data = scale.fit_transform(eeg_data)
    return eeg_data


def impute():
    pass

def remove_channels():
    pass


def padding():
    pass


def resampling():
    # Mario's code optimized with MNE

    # epoch the data before resampling otherwise

    # mneresample()
    pass

def remove_outliers(eeg_data: np.array):
    """
    Return a numpy array of EEG data without arterfacts.
    """

    # remove artefacts

    # from MNE docu: frequency-restricted artifacts are slow drifts and power line noise
    # power line noise -> notch filter

    # slow-drift



if "__main__" == __name__:
    n_freqs = 643
    seconds = 15
    total_samples = n_freqs*seconds
    raw_eeg = np.linspace(50,total_samples+100,n_freqs)
    print(n_freqs)
    print(raw_eeg)
    print(len(raw_eeg))
    print()
    resampled_data = resample_eeg_data(raw_eeg, n_freqs,100)
    print(resampled_data)
    print(len(resampled_data))
