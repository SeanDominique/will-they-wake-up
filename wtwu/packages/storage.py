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


            ################# OLD ######################
            # ##### TESTING
            # order = "001"
            # hour = "004"
            # #####


            # # .txt file with patient info
            # patient_data_filepath = os.path.join(PATIENT_DATA_PATH, patient_id, f"{patient_id}.txt")
            # # .hea file with info about a single EEG recording
            # eeg_header_filepath = os.path.join(PATIENT_DATA_PATH, patient_id, f"{patient_id}_{order}_{hour}_EEG.hea")
            # # .mat file with raw EEG data
            # eeg_data_filepath = os.path.join(PATIENT_DATA_PATH, patient_id, f"{patient_id}_{order}_{hour}_EEG.mat")
            # # eeg_data_filepath = os.path.join(PATIENT_DATA_PATH, patient_id, f"{patient_id}_{order}_{hour}_EEG.mat")

            # for loop to
                # download/stream eeg data (.mat) and eeg header (.hea) from GCS
                # extract the data into a "manipulate-able" form
            # ---> gcs_wtwu_blobs = gcs.list_blobs(BUCKET_NAME)

            ################# OLD ######################



            ##### TESTING
            # print(".hea content:         ", hea_file_content) # lines from .hea file
            #     # first line
            #         #0284_001_004_EEG 19 500 1578500
            #     # example middle lines
            #         #0284_001_004_EEG.mat 16+24 17.980017665549088 16 23877 24177 37933865398 0 Fp1
            #     # last three lines
            #         #Utility frequency: 50
            #         #Start time: 4:07:23
            #         #End time: 4:59:59
            # print(".hea content type:    ", type(hea_file_content)) # <class 'str'>
            # print()

            # print(".mat content:         ", eeg_file_content) # bunch of bytes
            # print(".mat content type:    ", type(eeg_file_content)) # <class 'bytes'>
            # print()
            ######


            ### debugging
            # temp = "gs://data-wtwa/i-care-2.0.physionet.org/training/0284/0284.txt"
            # temp = "i-care-2.0.physionet.org/training/0284/0284.txt"
            # data-wtwa/gs://data-wtwa/i-care-2.0.physionet.org/training/0284/0284.txt
            # https://storage.googleapis.com/download/storage/v1/b/data-wtwa/o/gs%3A%2F%2Fdata-wtwa%2Fi-care-2.0.physionet.org%2Ftraining%2F0284%2F0284.txt?alt=media
            ###

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
    print(all_eeg_data)


# def preprocess(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
#     """
#     - Query the raw dataset from Le Wagon's BigQuery dataset
#     - Cache query result as a local CSV if it doesn't exist locally
#     - Process query data
#     - Store processed data on your personal BQ (truncate existing table if it exists)
#     - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
#     """

#     print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

#     # Query raw data from BigQuery using `get_data_with_cache`
#     min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
#     max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

#     query = f"""
#         SELECT {",".join(COLUMN_NAMES_RAW)}
#         FROM `{GCP_PROJECT_WAGON}`.{BQ_DATASET}.raw_{DATA_SIZE}
#         WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
#         ORDER BY pickup_datetime
#     """

#     # Retrieve data using `get_data_with_cache`
#     data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
#     data_query = get_data_with_cache(
#         query=query,
#         gcp_project=GCP_PROJECT,
#         cache_path=data_query_cache_path,
#         data_has_header=True
#     )

#     # Process data
#     data_clean = clean_data(data_query)

#     X = data_clean.drop("fare_amount", axis=1)
#     y = data_clean[["fare_amount"]]

#     X_processed = preprocess_features(X)

#     # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
#     # using data.load_data_to_bq()
#     data_processed_with_timestamp = pd.DataFrame(np.concatenate((
#         data_clean[["pickup_datetime"]],
#         X_processed,
#         y,
#     ), axis=1))

#     load_data_to_bq(
#         data_processed_with_timestamp,
#         gcp_project=GCP_PROJECT,
#         bq_dataset=BQ_DATASET,
#         table=f'processed_{DATA_SIZE}',
#         truncate=True
#     )

#     print("✅ preprocess() done \n")
