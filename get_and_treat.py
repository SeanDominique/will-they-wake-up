import wtwu.packages.storage as wtstorage
import wtwu.packages.data as wtdata
import os
import numpy as np
import pickle


training_folder = "/home/mario/code/SeanDominique/will-they-wake-up/data/physionet.org/files/i-care/2.1/training"
#list_of_patients = ['0931', '0826', '0693']
list_of_patients = ([name for name in os.listdir(training_folder) if name.isnumeric()])

for patient in list_of_patients:
    survived, eeg_data_headers, all_eeg_data = wtstorage.import_data(patient)

    fs = eeg_data_headers[0]['fs']
    if not os.path.exists(f'./data/processed/{patient}'):
        os.makedirs(f'./data/processed/{patient}')
    with open(f'./data/processed/{patient}/y.txt', 'a+') as f:

        f.write(f'survived:{survived}\n')

    reduced_eeg_data = wtdata.reduce_all_channels(all_eeg_data,
                                         target_freq= 100,
                                         original_freq=
                                         fs
                                         )

    list_of_splits, list_of_times = wtdata.sample_all(reduced_eeg_data,hours=None)
    psds_fs, list_of_psds = wtdata.get_psds(list_of_splits)
    with open(f'./data/processed/{patient}/psds.npy', 'wb') as f:

        np.save(f, list_of_psds)
    with open(f'./data/processed/{patient}/psds_fs.npy', 'wb') as f:

        np.save(f, psds_fs)

    with open(f'./data/processed/{patient}/time_splits.npy', 'wb') as f:

        np.save(f, list_of_splits)

    with open(f'./data/processed/{patient}/headers.txt', 'wb') as f:
        pickle.dump(eeg_data_headers, f)

    print(patient, ' done!')
