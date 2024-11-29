import wtwu.packages.storage as wtstorage
import wtwu.packages.data as wtdata
import os
import numpy as np
import pickle
import sys

#list_of_patients = ([name for name in os.listdir(training_folder) if name.isnumeric()])
patient = sys.argv[1] # input string or int in CLI, assumes 0 at the start of the patient number if < 1000
#for patient in list_of_patients:
if not os.path.exists(f'./data/processed/{patient}/headers.pkl'):

    survived, eeg_data_headers, all_eeg_data = wtstorage.import_data(patient)

    # in case no relevant data is found from import_data
    if eeg_data_headers != 'Error' and len(eeg_data_headers)>0:

        fs = eeg_data_headers[0]['fs']
        if not os.path.exists(f'./data/processed/{patient}'):
            os.makedirs(f'./data/processed/{patient}')

        # .txt file containing True/False based on if this patient survived
        with open(f'./data/processed/{patient}/y.txt', 'a+') as f:

            f.write(f'survived:{survived}\n')

        # undersamples to 100Hz
        reduced_eeg_data = wtdata.reduce_all_channels(all_eeg_data,
                                            target_freq= 100,
                                            original_freq=fs
                                            )

        # splits TS into 15 second windows every 10 minutes
        list_of_splits, list_of_times = wtdata.sample_all(reduced_eeg_data,hours=None)

        # creates PSDs of these 15s windows
        psds_fs, list_of_psds = wtdata.get_psds(list_of_splits)


        # .npy file containing PSDs for each observation of this patient
        # number of psds = number of >1h long EEG recordings * 6 (for each 10 minute segment)
        with open(f'./data/processed/{patient}/psds.npy', 'wb') as f:
            np.save(f, list_of_psds)

        # psds_fs.psds contains the x-axis for the PSDs
        with open(f'./data/processed/{patient}/psds_fs.npy', 'wb') as f:
            np.save(f, psds_fs)

        # time_splits.npy contains arrays of 3 dimensions (X, Y, Z)
        # X = number of 15s segments collected from the patients EEG TS
        # Y = 1500 ( = 15s * 100 Hz)
        # Z = number of channels (ie. 8)
        with open(f'./data/processed/{patient}/time_splits.npy', 'wb') as f:
            np.save(f, list_of_splits)

        # headers.pkl contains a list with all the headers for each array in time_splits.npy
        with open(f'./data/processed/{patient}/headers.pkl', 'wb') as f:
            pickle.dump(eeg_data_headers, f)

        print(patient, ' done!')
    else:
        print(patient, ' skipped!')
    del survived
    del eeg_data_headers
    del all_eeg_data
else:
    print(patient, ' already present!')
