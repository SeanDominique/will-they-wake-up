import scipy.io
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
import os
from scipy.signal import welch
import os
import numpy as np
from google.cloud import storage



file_path = '/home/mariorocha/code/SeanDominique/will-they-wake-up/data/raw/physionet.org/files/i-care/2.1/training/0284/'
##TODO Change the path using os, if possible making the patient a variable.

def check_outcome(file_path):
    """
    Reads a file and returns 1 if the Outcome is 'Good', otherwise 0.

    Args:
        file_path (str): Path to the file containing patient data.

    Returns:
        int: 1 if Outcome is 'Good', 0 otherwise.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Outcome:"):
                outcome = line.split(":")[1].strip()
                return 1 if outcome == "Good" else 0
    # Return 0 if 'Outcome' line is missing
    return 0


def parse_eeg_file(file_path):

    '''get info from EEG.hea files.'''
    result = {
        "first_line_numbers": [],
        "matching_lines": {}
    }

    search_strings = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Process the first line
        first_line_parts = lines[0].strip().split()
        result["first_line_numbers"] = list(map(int, first_line_parts[-3:]))

        # Search for specific strings in subsequent lines
        for search_string in search_strings:
            for line_index, line in enumerate(lines[1:], start=2):  # Start from 2 since the first line is processed
                if search_string in line:
                    result["matching_lines"][search_string] = line_index
                    break  # Stop searching after the first match

    return result




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
    outcome = check_outcome(patient+patient[-5:]+'.txt')
    EEG_file_list = get_eeg_paths(patient)
    hours =[]

    if scaler == 'MinMax':
        scaler = MinMaxScaler()
    if scaler == 'Standard':
        scaler = StandardScaler()
    if scaler == 'Robust':
        scaler = RobustScaler()

    EEG_list = []
    headers_list = []


    for file_path in EEG_file_list[1:-1]:

        eeg = scipy.io.loadmat(file_path)
        header = parse_eeg_file(file_path[:-3]+'hea')
        # for line in eeg:
        #     temp_line = line.reshape(1, -1)
        #     temp_line = scaler.fit_transform(temp_line)
        #     line = temp_line.reshape(-1)
        # eeg = np.mean(eeg,axis=0)

        if header["first_line_numbers"][2]/header["first_line_numbers"][1] == 3600:
            eeg = [eeg['val'][header["matching_lines"]["Fp1"]-2],
                   eeg['val'][header["matching_lines"]["Fp2"]-2],
                   eeg['val'][header["matching_lines"]["F3"]-2],
                   eeg['val'][header["matching_lines"]["F4"]-2],
                   eeg['val'][header["matching_lines"]["C3"]-2],
                   eeg['val'][header["matching_lines"]["C4"]-2],
                   eeg['val'][header["matching_lines"]["P3"]-2],
                   eeg['val'][header["matching_lines"]["P4"]-2],
                   ]
            for line in eeg:
                temp_line = line.reshape(1,-1)
                temp_line = scaler.fit_transform(temp_line)
                line = temp_line.reshape(-1)
            eeg = np.array(eeg)
            eeg = eeg.astype(float)

            headers_list.append(header)

            EEG_list.append(eeg)
            hour = (file_path[-11:-8])
            print(hour)
            hours.append(hour)

    freq = headers_list[0]["first_line_numbers"][1]

    EEG_list = np.array(EEG_list)
    hours = np.array(hours).astype(np.float16)


    return EEG_list, outcome, freq, hours

def reduce_EEGs(list_of_EEGs, target_freq = 100, original_freq = 500):
    '''
    This function takes a list of EEG spectra, the target frequency
    and the original_frequency, the last two defaulted at 100 and 500.


    '''


    #rate_of_reduction = np.round(rate_of_reduction)


    def resample_time_series_array(time_series, original_freq, target_freq):
        """
        Resample a time series to a target frequency, even if it's not a divisor of the original frequency.

        Args:
            time_series (np.ndarray): Original time series as a 1D numpy array.
            original_freq (float): Original frequency in Hz.
            target_freq (float): Target frequency in Hz.

        Returns:
            np.ndarray: Resampled time series.
            np.ndarray: New time points corresponding to the resampled series.
        """
        # Original time points
        original_time_points = np.arange(len(time_series)) / original_freq

        # Target time points
        duration = original_time_points[-1]  # Total duration in seconds
        target_time_points = np.arange(0, duration, 1 / target_freq)

        # Interpolation
        interpolated_values = np.interp(
            target_time_points,
            original_time_points,
            time_series
        )

        return interpolated_values, target_time_points





    new_list_of_EEGs = []
    for EEG in list_of_EEGs:
        EEG , times = resample_time_series_array(EEG,original_freq,target_freq)
        new_list_of_EEGs.append(EEG)
    new_list_of_EEGs = np.array(new_list_of_EEGs)
    return new_list_of_EEGs

def reduce_all_channels(channels_array, target_freq = 100, original_freq =500):
    #rate_of_reduction = np.round(rate_of_reduction)

    reduced_channels = []
    for i in range(0,channels_array.shape[0]):
        reduced_channels.append(reduce_EEGs(channels_array[i,:,:],
                                              target_freq= target_freq,
                                              original_freq=original_freq
                                              ))
    reduced_channels = np.array(reduced_channels)
    return reduced_channels

def sampling_EEGs(list_of_EEGs, fs=100, sampling_rate=600, sampling_size=15,hours=np.zeros((10000,))):
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


    for EEG in list_of_EEGs:
        i = 0


        while ((sampling_rate*i) + sampling_size)*fs < len(EEG):
            #print((len(EEG),((sampling_rate*i) + sampling_size)*fs), 'print 1')

            splits.append(EEG[(sampling_rate*i)*fs:((sampling_rate*i)+sampling_size)*fs])

            if hours.shape != (10000,):
                #print((hours), 'print 2')

                split_time.append((hours)+((fs/sampling_rate)*i))
                #print(hours,(hours)+((fs/sampling_rate)*i), 'print 3')

            i += 1






    splits = np.array(splits)

    if len(split_time)>0:
        split_time = np.array(split_time)
        print(splits.shape, split_time.shape)

    return splits, split_time

def sample_all(reduced_array, fs=100, sampling_rate=600, sampling_size=15,hours=None):
    '''Does the sampling in all channels and all times'''
    list_of_splits = []
    list_of_split_times = []
    for i in range(0,reduced_array.shape[0]):
        split, split_time = sampling_EEGs(reduced_array[i,:,:],hours=hours[i])
        #print(split_time[:int(sampling_rate/fs)])
        list_of_splits.append(split)
        list_of_split_times.append(split_time[:int(sampling_rate/fs)])

    list_of_splits = np.array(list_of_splits)

    list_of_splits = list_of_splits.reshape(int(list_of_splits.shape[0]),8,int(list_of_splits.shape[1]/8),int(list_of_splits.shape[2]))
    list_of_splits = np.transpose(list_of_splits,axes=(0,2,3,1))
    list_of_splits = list_of_splits.reshape(list_of_splits.shape[0]*list_of_splits.shape[1],list_of_splits.shape[2],list_of_splits.shape[3])

    if len(list_of_split_times)>0:
        print(list_of_split_times)

        list_of_split_times = np.concatenate(list_of_split_times)
        print(list_of_split_times)
        #list_of_split_times = list_of_split_times[np.mod(np.arange(list_of_split_times.size), 8) != 0]
        #print(list_of_split_times.shape)

        #list_of_split_times = list_of_split_times[:,0]

    return list_of_splits, list_of_split_times



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



    if input_type == 'list':
        psds = []
        for eeg in EEG_list:
            f, temp_psd = welch(eeg, fs=fs, nperseg=1024)
            psds.append(temp_psd)
        psds_df = pd.DataFrame(psds)
        if mode == 'time' and hours.shape[1] == psds_ar.shape[1]:
            psds_df = pd.concat([psds_df, hours], axis=1)
            psds_df = psds_df.groupby(by='hours',as_index=True).mean()
            return f, psds_df
        elif mode == 'time':
            return f, psds_df
        elif mode == 'channels':
            return f, psds_df
        else:
            print('get_psds: unrecognised mode.')
            return None

    if input_type == 'array':
        psds = []
        for i in range(0, EEG_list.shape[0]):
            psds1 = []
            for j in range(0, EEG_list.shape[2]):
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
            print('get_psds: unrecognised mode.')
            return None

    else:
        print('get_psds: bad input type')
        return None



client = storage.Client()

def list_gcs_files(bucket_name, prefix):
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs]

def validate_patient_data(bucket_name, prefix, patients):
    """
    Vérifie que chaque patient a les fichiers nécessaires sur GCP et signale les données manquantes.
    Si le dossier d'un patient est absent, il est ignoré.
    """
    required_files = ["psds.npy", "times.npy", "y.txt"]
    files_on_gcs = list_gcs_files(bucket_name, prefix)  # Liste complète des fichiers présents
    missing_data = []

    for patient in patients:
        # Formater l'ID du patient avec des zéros (toujours 4 chiffres)
        patient_id = f"{int(patient):04d}"
        patient_prefix = f"{prefix}/{patient_id}"

        # Vérifier si le dossier du patient existe
        if not any(file.startswith(patient_prefix) for file in files_on_gcs):
            print(f"Dossier manquant pour le patient {patient_id}, passage au suivant.")
            continue

        # Vérifier la présence des fichiers nécessaires
        for file_name in required_files:
            expected_file = f"{patient_prefix}/{file_name}"
            if expected_file not in files_on_gcs:
                missing_data.append((patient_id, expected_file))

    if len(missing_data) > 0:
        print("Données manquantes :")
        for patient_id, file_path in missing_data:
            print(f"Patient {patient_id}, Fichier manquant : {file_path}")
        return False
    else:
        print("Toutes les données nécessaires sont présentes.")
        return True

def create_batches(data, labels, batch_size):
    """
    Découpe les données et les labels en batchs de taille fixe.
    """
    num_batches = len(data) // batch_size
    data_batches = [
        data[i * batch_size:(i + 1) * batch_size]
        for i in range(num_batches)
    ]
    label_batches = [
        labels[i * batch_size:(i + 1) * batch_size]
        for i in range(num_batches)
    ]

    # Si des données restantes, créer un dernier batch
    if len(data) % batch_size != 0:
        data_batches.append(data[num_batches * batch_size:])
        label_batches.append(labels[num_batches * batch_size:])

    print(f"Nombre de batchs : {len(data_batches)}")
    return data_batches, label_batches



def blob_exists(bucket, blob_name):
    """
    Vérifie si un fichier existe dans le bucket GCP.
    """
    blob = bucket.blob(blob_name)
    return blob.exists()

import numpy as np

def create_global_dataset(bucket_name, prefix, patients):
    """
    Charge les données `time_splits` et les labels `y` pour tous les patients,
    et les concatène en un dataset global.
    """
    all_time_splits = []
    all_labels = []
    bucket = client.bucket(bucket_name)

    for patient in patients:
        try:
            # Formater l'ID du patient avec des zéros
            patient_id = f"{int(patient):04d}"
            patient_prefix = f"{prefix}/{patient_id}"

            # Charger les fichiers nécessaires depuis GCP
            time_splits_blob = bucket.blob(f"{patient_prefix}/time_splits.npy")
            y_blob = bucket.blob(f"{patient_prefix}/y.txt")

            time_splits_local = f"./temp/{patient_id}_time_splits.npy"
            y_local = f"./temp/{patient_id}_y.txt"

            time_splits_blob.download_to_filename(time_splits_local)
            y_blob.download_to_filename(y_local)

            # Charger les données localement
            time_splits = np.load(time_splits_local)
            with open(y_local, "r") as f:
                raw_label = f.readline().split(":")[1].strip()
                label = 1 if raw_label.lower() == "true" else 0

            # Ajouter les données au dataset global
            all_time_splits.append(time_splits)
            all_labels.extend([label] * len(time_splits))

            # Nettoyer les fichiers locaux
            os.remove(time_splits_local)
            os.remove(y_local)

        except Exception as e:
            print(f"Erreur pour le patient {patient_id} : {e}")

    # Concaténer toutes les données
    all_time_splits = np.concatenate(all_time_splits, axis=0)
    all_labels = np.array(all_labels)

    print(f"Dataset global créé : {all_time_splits.shape}, {all_labels.shape}")
    return all_time_splits, all_labels

def create_time_dependent_dataset(bucket_name, prefix, patients, initial_time=0,end_time=1000):
    """
    Charge les données `time_splits` et les labels `y` pour tous les patients,
    et les concatène en un dataset global.
    """
    all_time_splits = []
    all_labels = []
    bucket = client.bucket(bucket_name)

    for patient in patients:
        try:
            # Formater l'ID du patient avec des zéros
            patient_id = f"{int(patient):04d}"
            patient_prefix = f"{prefix}/{patient_id}"

            # Charger les fichiers nécessaires depuis GCP
            time_splits_blob = bucket.blob(f"{patient_prefix}/time_splits.npy")
            y_blob = bucket.blob(f"{patient_prefix}/y.txt")

            time_splits_local = f"./temp/{patient_id}_time_splits.npy"
            y_local = f"./temp/{patient_id}_y.txt"

            time_stamps_local = f"./temp/{patient_id}_times.npy"

            time_splits_blob.download_to_filename(time_splits_local)
            y_blob.download_to_filename(y_local)

            # Charger les données localement
            time_splits = np.load(time_splits_local)
            print(time_splits.shape)
            time_stamps = np.load(time_stamps_local)
            print(time_stamps)
            mask = (time_stamps >= initial_time) & (time_stamps <= end_time)
            print(mask.shape)
            time_splits = time_splits[mask]


            with open(y_local, "r") as f:
                raw_label = f.readline().split(":")[1].strip()
                label = 1 if raw_label.lower() == "true" else 0

            # Ajouter les données au dataset global
            if len(time_splits) > 0:

                all_time_splits.append(time_splits)
                all_labels.extend([label] * len(time_splits))
            else:
                print(f"Patient {patient_id} : pas de données entre {initial_time} et {end_time} heures.")
            # Nettoyer les fichiers locaux
            os.remove(time_splits_local)
            os.remove(y_local)
            os.remove(time_stamps_local)


        except Exception as e:
            print(f"Erreur pour le patient {patient_id} : {e}")

    # Concaténer toutes les données
    all_time_splits = np.concatenate(all_time_splits, axis=0)
    all_labels = np.array(all_labels)

    print(f"Dataset global créé : {all_time_splits.shape}, {all_labels.shape}")
    return all_time_splits, all_labels
