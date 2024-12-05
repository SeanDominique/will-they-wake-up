from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler

from wtwu.params import *
from wtwu.packages.storage import get_list_of_patients, import_data, upload_preprocessed_data_to_gcs

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

    try:
        # get patient data from GCS
        if DATA_TARGET == "gcs":
            if len(patients) == 0:
                patients = get_list_of_patients()

            for patient in patients:
                print(f"Traitement du patient {patient}...")

                # Import des données
                survived, eeg_data_headers, all_eeg_data = import_data(patient)
                print("all_eeg_data shape: ", all_eeg_data.shape)

                if eeg_data_headers != "Error" and len(eeg_data_headers) > 0:

                    fs = eeg_data_headers[0]['fs']
                    hours = np.array([header['recording_hour'] for header in eeg_data_headers]).astype(np.float16)

                    # Réduction des données
                        # TODO: try undersampling of 125Hz and 128Hz (in research paper)
                    print("undersampling")
                    undersampled_eeg_data = undersample_eegs(all_eeg_data, new_freq=100, original_freq=fs)
                    print("underdsampled_eeg_data shape: ", undersampled_eeg_data.shape)

                    # Re-reference (mean, local, outter electrode...)


                    # Bandpass filter (0.1-40Hz)
                    # `filter_data`` default uses FIR method which is more computationally intensive, but more stable. Using this over butterworth filter, an IIR method, because we are more focused on temporal analysis where phase distortion could be a problem.
                    print("bandpass filter")
                    bandpassed_eeg_data = filter_data(undersampled_eeg_data,
                                                    sfreq=fs,
                                                    l_freq=0.01,
                                                    h_freq=40)
                    print("bandpass filtered shape: ", bandpassed_eeg_data.shape)

                    # Notch filter
                        # TODO: Need to check if literature suggests to apply notch filter even if undersampling AND applying bandpass filter to < utility freq (my assumption is no because it was removed during the bandpass, although depending on the order of the butterworth filter "residue" frequency could be present)
                    # utility_freq = round(int(eeg_data_headers[0]['utility_freq']))
                    # if utility_freq == 50:
                    #     print("notch filter")
                    #     notched_eeg_data = notch_filter(bandpassed_eeg_data,
                    #                                     Fs=100,
                    #                                     freqs=utility_freq
                    #                                     )

                    # Artifacts: removal and imputation [advanced]


                    # ICA


                    # Epoching
                    # # Fenêtrage et normalisation
                    print("epoching")
                    eeg_epochs, split_times = epoch_eeg(bandpassed_eeg_data, hours=hours)


                    # Artifacts: Remove epochs with artefacts [simplistic]\
                    print("artefact removal")
                    indeces = []
                    for i, epoch in enumerate(eeg_epochs):
                        if remove_artifacts(epoch):
                            indeces.append(i)
                    np.delete(eeg_epochs, indeces)


                    # Data imputation
                        # TODO: calculate the number of flatlines + reexplore bad channels through EDA


                    # Waveform segmentation using rolling window


                    # standardize (z-score normalization)
                    print("normalize")
                    standardized_eeg_epochs = standardize(eeg_epochs)
                    print("standardized_eeg shape", standardized_eeg_epochs.shape)

                    # calcul des PSD
                        # for feature engineering or other models
                    print("creating PSDs")
                    f_list, psds_list = get_psds(standardized_eeg_epochs)

                    ### Preprocessing patient data DONE
                    print("Data preprocessing done for patient: {patient}")

                    # upload patient info to relevant GCS bucket
                    patient_info = {
                        "PSDs.npy": psds_list,
                        "PSDs_fs.npy": f_list,
                        "time_splits.npy": standardized_eeg_epochs,
                        "times.npy": split_times,
                        "header.pkl": eeg_data_headers
                    }
                    upload_preprocessed_data_to_gcs(patient, patient_info, survived)


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

    except Exception as e:
        print(f"Couldn't preprocess data correctly due to error: {e}")

    return

############ Preprocessing helper functions ############

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


    # convert to dtype that mne.resample() can handle
    converted_eeg = raw_eeg.astype(np.float64)

    resampling_ratio = original_freq / new_freq

    # uses anti-aliasing,
    # the lower frequencies we are interested in [0.1 - 40]Hz are safe
    # because the Nyquist frequency is new_freq/2 = 50 Hz.
    # Therefore the low pass filter used to avoid aliasing won't cut out the frequencies of interest.
    # note: the Nyquist frequency is dependent on the new frequency -> this is because undersampling implies
    #       you are using a lower frequency to capture the same signal, so higher frequency components will be lost
    undersampled_eeg_data = resample(converted_eeg, down=resampling_ratio)



    # in case resampled array is too large
    max_samples = max_seconds * new_freq
    if len(undersampled_eeg_data) > max_samples:
        remaining = int(len(undersampled_eeg_data) % resampling_ratio)
        undersampled_eeg_data = undersampled_eeg_data[:max_samples]
        print(f'Data points lost: {remaining}')

    # print(f'EEG reduced to a {original_freq/resampling_ratio} Hz frequence.')

    return undersampled_eeg_data

### ✅ Epoching
def sampling_EEGs(arr_of_eegs, fs=100, sampling_rate=600, sampling_size=15, hours=np.zeros((10000,))):
    '''
    This function takes a numpy array of numpy arrays containing a patient's EEG data for each 1h of recording.

    It also takes their frequency, the sampling rate in seconds (every 10 min = 600), the sampling size in seconds,
    and if available, the list of hours after cardiac arrest.

    It returns splits of the EEGs every (sampling_rate) seconds of length (sampling_size)
    seconds. If a list of hours was given, it also returns a list of decimal times in hours.
    This list can later be transformed in HH:MM easily, if needed.

    Parameters:
    - list_of_EEGs (np.array): array of all a patient's EEG data
    - fs (int):                sampling frequency of EEG data (after undersampling in `undersample_eegs`)
    - sampling_rate (int):     the interval, in seconds, at which to take the epochs
    - sampling_size (int):     number of seconds for the epoch
    - hours :

    Returns:
    - eeg_epochs (np.array): array of EEG epochs (shape: (sampling_size * fs,))
    - time_splits (np.array):
    '''

    # TODO: use mne.make_fixed_length_epochs, note: takes `Raw` object instead of array-like signal

    eeg_epochs = []
    split_time = []

    for i_eeg, eeg in enumerate(arr_of_eegs):
        i = 0
        while ((sampling_rate*i) + sampling_size) * fs < len(eeg):
            eeg_epochs.append(eeg[(sampling_rate*i)*fs:((sampling_rate*i)+sampling_size)*fs])

            # to check if `hours` corresponds to an actual list of hours
            if hours.shape != (10000,):
                # 0/6 -> 1/6 -> 2/6 ... -> 5/6
                split_time.append((hours)+((fs/sampling_rate)*i)) # TODO: Ask Mario about `i_eeg`
            i += 1

    eeg_epochs = np.array(eeg_epochs)

    if len(split_time)>0:
        split_time = np.array(split_time)

    return np.array(eeg_epochs), split_time

def epoch_eeg(arr_reduced_eeg, fs=100, sampling_rate=600, sampling_size=15,hours=None):
    '''
    Does the sampling in all channels and all times
    undersampled EEG data -> 15s epochs
    TODO: rewrite this function doc
    '''

    eeg_epochs = []
    split_times = []

    print(arr_reduced_eeg.shape)
    for i in range(arr_reduced_eeg.shape[0]):
        epochs, split_time = sampling_EEGs(arr_reduced_eeg[i,:,:], hours=hours[i])
        eeg_epochs.append(epochs)

        # to avoid copying split_time 6 times per EEG 1h recording
        split_times.append(split_time[:int(sampling_rate/fs)])

    # reshape epochs for model input_shape
    eeg_epochs = np.array(eeg_epochs)
    eeg_epochs = eeg_epochs.reshape(int(eeg_epochs.shape[0]),8,int(eeg_epochs.shape[1]/8),int(eeg_epochs.shape[2]))
    eeg_epochs = np.transpose(eeg_epochs,axes=(0,2,3,1))
    eeg_epochs = eeg_epochs.reshape(eeg_epochs.shape[0]*eeg_epochs.shape[1],eeg_epochs.shape[2],eeg_epochs.shape[3])

    if len(split_times)>0:
        split_times = np.concatenate(split_times)
        # split_times = np.array(split_times)
        # split_times = split_times[0,:]

    return eeg_epochs, split_times

### PSDS, frequency domain
def get_psds(EEG_list,fs=100, mode='channels', hours=np.zeros((2,3,4,5,5)),input_type='array'):
    '''
    This function takes an array or list of EEGs and returns the PSDs.

    It can have two modes: 'channels' or 'time':

    On 'channels' mode, it will return a PSD for each channel
    if given a list of EEGs of a single patient from a single raw file.

    On 'time' mode, it will take as input a list containing EEGs of a single
    channel or an average of a single patient and return a dataframe containing
    all PSDs as columns and their 'hours after cardiac arrest' as index.

    input_type = ['list' ,'array']
    '''

    psds = []
    print("psd-ing")
    if not mode in ["channels", "time"]:
        print('get_psds: unrecognised mode.')
        return None

    if input_type == 'list':

        for eeg in EEG_list:
            f, temp_psd = welch(eeg, fs=fs, nperseg=1024)
            psds.append(temp_psd)

        psds_df = pd.DataFrame(psds)


        if mode == 'time':
            if hours.shape[1] == psds_ar.shape[1]:
                psds_df = pd.concat([psds_df, hours], axis=1)
                psds_df = psds_df.groupby(by='hours',as_index=True).mean()
            return f, psds_df

        elif mode == 'channels':
            return f, psds_df


    if input_type == 'array':
        print("EEG_list shape: ", EEG_list.shape)
        for i in range(EEG_list.shape[0]):
            psds1 = []
            for j in range(EEG_list.shape[2]):
                f, psds_temp = welch(EEG_list[i,:,j], fs=fs, nperseg=4*fs)
                psds1.append(psds_temp)
            psds.append(psds1)

        psds_ar = np.array(psds)
        psds_ar = np.transpose(psds_ar, axes=(0,2,1))

        if mode == 'time':
            psds_df = pd.DataFrame(psds_ar)
            if hours.shape[0] == psds_ar.shape[1]:
                psds_df = pd.concat([psds_df, hours], axis=1)
                psds_df = psds_df.groupby(by='hours',as_index=True).mean()
                return f, psds_df
            return f, psds_df

        elif mode == 'channels':
            return f, psds_ar

    else:
        print('get_psds: bad input type')
        return None


### ✅ Data manipulation
def standardize(eeg_data: np.array, scaler="standard") -> np.array:
    """
    Return a numpy array with the standardized EEG data based on the given scaler
    """
    print("standaaaaardizing")
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

    scaled_eeg_data = []
    for epoch in eeg_data:
        flat_epoch = epoch.flatten()
        scaled_flattened = scale.fit_transform(flat_epoch.reshape(-1, 1)).flatten()
        scaled_epoch = scaled_flattened.reshape(epoch.shape)
        scaled_eeg_data.append(scaled_epoch)

    return np.array(scaled_eeg_data)


def impute():
    pass

def remove_channels():
    # TODO: note: we only collect relevant channels in `import_data`
    # look into MNE raw["bads"] -> how do they determine bad channels?
    pass

def padding():
    # unnecessary considering current model (LSTM) input shape
    pass


def remove_artifacts(eeg_epoch: np.array, threshold=100):
    """
    [Simplistic approach]
    Evaluates if an eeg_epoch contains artefacts.

    [Further development]
    Return a numpy array of EEG data without artifacts.

    From MNE doc, frequency-restricted artifacts are slow drifts and power line noise
    # power line noise -> notch filter
    # slow-drift -> # TODO: requires EDA
    """

    # remove ECG signal
        # TODO: import ECG signals from GCS and use in preprocessing step

    # remove excessively high amplitude signals

    # slow-drift

    return (np.max(np.abs(eeg_epoch)) > threshold) # indicate if EEG signal amplitude (mV) is higher than threshold


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
