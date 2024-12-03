# GOAL: GET RAW DATA FROM BUCKET INTO A SCRIPT SO THAT
# A/ we can process it
# B/ we can move data to bq
# C/ we can call BQ to visualize the data

# the person who wants to work with the data should be able to:
# access the data from any patient they want


######## IMPORTS #######
import os
from wtwu.params import *
import re
import pickle

from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor

from scipy.io import loadmat
import numpy as np
import pandas as pd
import io
from google.cloud import bigquery
# # to automate with model lifecycle
# import mlflow
# import pickle

######## CONSTANTS ########
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"] # channels we are interested in

def get_list_of_patients():
    """
    Récupère la liste des patients à partir des blobs présents dans le bucket Google Storage.
    """

    client = storage.Client()
    blobs = client.list_blobs(BUCKET_NAME, prefix=f"{PATIENT_DATA_PATH}", delimiter='/')

    # Extraire les ID des patients
    patient_ids = set()
    for blob in blobs:
        path_parts = blob.name.split("/")
        if len(path_parts) > 2 and path_parts[-2].isnumeric():
            patient_ids.add(path_parts[-2])

    return sorted(list(patient_ids))


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
    survived = 0
    eeg_data_headers = []
    all_eeg_data = []
    # TODO: Start / connect to VM


    # import patient data
    if DATA_TARGET == "gcs":

        print("importing data from GCS...")

        try:
            # TODO: Threading to download blobs in parallel

            client = storage.Client()

            gcs_wtwu_blobs = client.list_blobs(BUCKET_NAME, prefix=f"{PATIENT_DATA_PATH}{patient_id}/", delimiter='/')

            # TODO: check if we actually need to collect multiple headers or if we can just use eeg_data.shape to check for sample length
            # # only collect the first header for a given patient
            # no_header = True
            eeg_data_headers = []
            all_eeg_data = []
            survived = None

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
                    hea_result = extract_header_data(hea_file_content, blob.name)
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

            print("data imported and returned")
            return  survived, eeg_data_headers, np.array(all_eeg_data)

        except Exception as e :

            print("Couldn't download from GCS", e)

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


def extract_header_data(header_file_content, file_name):
    """
    Takes the file content of a .hea as text and returns a dictionary with relevant values for EEG data preprocessing

    Returns:
        dict{
            "fs": # sampling frequency
            "num_channels": # number of total channels from recording
            "num_samples": # total number of samples in that .mat file
            "recording_hour": the hour the recording was made after ROSC (Return Of Spontaneous Circulation)
            "channels_index": {}, # dictionary with the indexes of each channel in the `eeg_data` numpy array
            "all_channels_found": False # True if the data contains values for all the channels in `CHANNELS`
        }
    """
    hea_result = {
        "fs": 0,
        "num_channels": 0,
        "num_samples": 0,
        "recording_hour": 0,
        "channels_index": {},
        "all_channels_found": False
    }

    file_hour = re.search(r'_(\d{3})_(\d{3})_', file_name)
    print(file_hour.group(2), file_name)
    hea_result["recording_hour"] = file_hour.group(2)

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



def upload_preprocessed_data_to_gcs(patient_id: str, patient_info: dict, survived: bool):
    """
    Upload les fichiers prétraités pour un patient vers le bucket GCP.

    Parameters:
    - patient_id (str):     id of the patient whose data is being uploaded to GCP [eg. "0284"]
    - patient_info (dict):  (keys: name of the file, values: arrays of EEG data or dictionary with header data)
    - survived (bool):      "True" or "False" depending on patient outcome
    """
    # TODO: multi-patient batch uploads

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    try:
        print("Uploading patient {patient} files to GCS.")
        # save numpy arrays as .npy and
        for name, info in patient_info.items():
            blob = bucket.blob(f"{PATIENT_PROCESSED_DATA_PATH}{patient_id}/{name}")
            buffer = io.BytesIO()
            if ".pkl" in name:
                pickle.dump(info, buffer)
            elif ".npy" in name:
                np.save(buffer, info)
            else:
                print("Unknown filetype given.")
                return
            buffer.seek(0)
            blob.upload_from_file(buffer, content_type="application/octet-stream")
            print(f"Successfully uploaded patient {patient_id} {name} file to GCS.")

        # save outcome label as .txt
        blob = bucket.blob(f"{PATIENT_PROCESSED_DATA_PATH}{patient_id}/y.txt")
        blob.upload_from_string(str(survived))
        print(f"Successfully uploaded patient {patient_id} y.txt file to GCS.")

    except Exception as e:
        print(f"Couldn't upload patient {patient_id} files to GCS. \nError: {e}")

    return



############### MODEL ###############
def save_metrics_to_bigquery(project_id, dataset_id, table_id, metrics):
    """
    Sauvegarde les métriques d'entraînement/validation dans une table BigQuery.
    """
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    # Insérer dans BigQuery
    errors = client.insert_rows_json(table_ref, metrics)
    if errors:
        print(f"Erreur lors de l'insertion des métriques : {errors}")
    else:
        print("Métriques sauvegardées dans BigQuery.")


def save_predictions_to_bigquery(project_id, dataset_id, table_id, predictions):
    """
    Sauvegarde les prédictions dans une table BigQuery.
    """
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    # Insérer dans BigQuery
    errors = client.insert_rows_json(table_ref, predictions)
    if errors:
        print(f"Erreur lors de l'insertion des prédictions : {errors}")
    else:
        print("Prédictions sauvegardées dans BigQuery.")



if __name__ == "__main__":
    survived, eeg_data_headers, all_eeg_data = import_data("0430")
    print(survived)
    print(eeg_data_headers)
    print(all_eeg_data)
