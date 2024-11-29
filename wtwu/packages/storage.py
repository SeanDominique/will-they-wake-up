# GOAL: GET RAW DATA FROM BUCKET INTO A SCRIPT SO THAT
# A/ we can process it
# B/ we can move data to bq
# C/ we can call BQ to visualize the data

# the person who wants to work with the data should be able to:
# access the data from any patient they want


######## IMPORTS #######
import os
from wtwu.params import *

from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
from scipy.io import loadmat
import numpy as np
import pandas as pd
import io

# # to automate with model lifecycle
# import mlflow
# import pickle

######## CONSTANTS ########
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"] # channels we are interested in


def import_data(patient_id: str):
    """
    Imports raw data for 1 patient from Google Cloud Storage (GCS)

    Parameters:
        patient_id : str --> Assumes `patient_id` includes a 0 at the start if patient number is < 1000. eg. "0284"

    Returns:
        survived : bool
        eeg_data_headers : list() --> contains dictionaries with EEG data array info
        all_eeg_data : np.array --> patient EEG data
    """

    # TODO: Start / connect to VM


    # import patient data
    if DATA_TARGET == "gcs":

        print("importing data from GCS...")

        try:
            # TODO: Threading to download blobs in parallel

            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)

            gcs_wtwu_blobs = client.list_blobs(BUCKET_NAME, prefix=f"{PATIENT_DATA_PATH}{patient_id}/", delimiter='/')

            # TODO: check if we actually need to collect multiple headers or if we can just use eeg_data.shape to check for sample length
            # # only collect the first header for a given patient
            # no_header = True
            eeg_data_headers = []
            all_eeg_data = []

            for blob in gcs_wtwu_blobs:
                # Download in memory to reduce I/O operations
                # Use `.download_as_text()` and `.download_as_bytes()`
                # (instead of using `.download_to_filename()` which downloads to disk)
                if blob.name.endswith(".txt"):
                    text_file_content  = blob.download_as_text()
                    # TODO: Extract other file contents from  .txt if we want to evolve model

                    with io.StringIO(text_file_content) as f:
                        lines = f.readlines()
                        # `Good` means the patient survived
                        if "Good" in lines[-2]:
                            survived = True
                        else:
                            survived = False

                elif blob.name.endswith("EEG.hea"):
                    hea_file_content  = blob.download_as_text()
                    hea_result = extract_header_data(hea_file_content)
                    eeg_data_headers.append(hea_result)


                elif blob.name.endswith("EEG.mat"):
                    # only take EEG recordings > 1h
                    # assumes that .hea files are processed before .mat in buckets
                    if (eeg_data_headers[-1]["num_samples"] / eeg_data_headers[-1]["fs"]) >= 3600:
                        eeg_file_content = blob.download_as_bytes()
                        eeg_data = extract_eeg_data(eeg_file_content, eeg_data_headers[-1])
                        all_eeg_data.append(eeg_data)
                    else:
                        # remove the last header
                        eeg_data_headers.pop()

            return  survived, eeg_data_headers, np.array(all_eeg_data)

        except:
            print("Couldn't download from GCS")
            return "Error", "Error", "Error"


    elif DATA_TARGET == "local":

        print("importing data from local...")
        # TODO: (low prio) logic to import from local machine
        pass

    else:
        print("env variables not connected")

    print("data imported")

    return survived, eeg_data_headers, np.array(all_eeg_data)



# TODO: implement checkpoints to avoid reprocessing in case of failures


def extract_header_data(header_file_content):
    """
    Takes the file content of a .hea as text and returns a dictionary with relevant values for EEG data preprocessing

    Returns:
        dict{
            "fs": # sampling frequency
            "num_channels": # number of total channels from recording
            "num_samples": # total number of samples in that .mat file
            "channels_index": {}, # dictionary with the indexes of each channel in the `eeg_data` numpy array
            "all_channels_found": False # True if the data contains values for all the channels in `CHANNELS`
        }
    """
    hea_result = {
        "fs": 0,
        "num_channels": 0,
        "num_samples": 0,
        "channels_index": {},
        "all_channels_found": False
    }

    with io.StringIO(header_file_content) as f:
        lines = f.readlines()

        # get info from current recording header
        first_line = lines[0].strip().split()

        hea_result["fs"] = int(first_line[2])
        hea_result["num_channels"] = int(first_line[1])
        hea_result["num_samples"] = int(first_line[3])

        # check which channels were recorded in the patient's dataset
        for channel in CHANNELS:
            for line_index, line in enumerate(lines[1:], start=2):  # Start from 2 since the first line are processed
                    if channel in line:
                        hea_result["channels_index"][channel] = line_index
                        break  # Stop searching after match


        if len(hea_result["channels_index"].keys()) == len(CHANNELS):
            hea_result["all_channels_found"] = True

    return hea_result


def extract_eeg_data(eeg_file_content, header):
    """
    Takes .mat file data as bytes and returns a np.array of dim = (n,m)
    n: number of channels
    m: number of samples
    """
    eeg_data = loadmat(io.BytesIO(eeg_file_content))
    eeg_data = eeg_data['val']

    # TODO: reshape the data to fit Mario's undersampling function in preprocess.py (code below)

    # -2 is an offset
    eeg_data_arr = np.array([
        eeg_data[header["channels_index"]["Fp1"]-2],
        eeg_data[header["channels_index"]["Fp2"]-2],
        eeg_data[header["channels_index"]["F3"]-2],
        eeg_data[header["channels_index"]["F4"]-2],
        eeg_data[header["channels_index"]["C3"]-2],
        eeg_data[header["channels_index"]["C4"]-2],
        eeg_data[header["channels_index"]["P3"]-2],
        eeg_data[header["channels_index"]["P4"]-2],
    ])

    return eeg_data_arr


def upload_processed_data_to_bq():
    """
    Upload preprocessed patient data to BQ DB.
    """
    # batch uploads
    # use BQ's streaming API for low-latency ingestion -> more costly than batch uploads
    pass



############### MODEL ###############
def load_model_from_gcs():
    pass


def save_model_to_gcs():
    pass


def save_results_to_gcs():
    pass


if __name__ == "__main__":
    survived, eeg_data_headers, all_eeg_data = import_data("0430")
    print(survived)
    print(eeg_data_headers)
    print(len(eeg_data_headers))
    print(all_eeg_data)
    print(all_eeg_data.shape)
