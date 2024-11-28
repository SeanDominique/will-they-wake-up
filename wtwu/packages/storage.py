# GOAL: GET RAW DATA FROM BUCKET INTO A SCRIPT SO THAT
# A/ we can process it
# B/ we can move data to bq
# C/ we can call BQ to visualize the data

# the person who wants to work with the data should be able to:
# access the data from any patient they want
import os
from wtwu.params import *

from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# # to automate with model lifecycle
# import mlflow
# import pickle



def import_data(patient_id: str):
    """
    Imports raw data for 1 patient from Google Cloud Storage (GCS)

    Parameters:
        patient_id (str) : Assumes `patient_id` includes a 0 at the start if patient number is < 1000. eg. "0284"
     Compile le modèle avec les paramètres spécifiés.

    Returns:
        patient EEG data
        patient info
    """
    # TODO: Start / connect to VM

    # import patient data
    if MODEL_TARGET == "gcs":

        print("importing data from GCS...")

        try:
            # in case there is no file in the bucket corresponding to the given patient_id

            # Threading to download blobs in parallel
            # Paginate through large buckets client.list_blobs("bucket_name", prefix="patient-files/")

            # if using distributed computing, use signed URLs to avoid needing to authenticate each time

            # .txt file with patient info
            patient_data_filepath = os.path.join(PATIENT_DATA_PATH, patient_id, f"{patient_id}.txt")

            # .hea file with info about a single EEG recording
            eeg_info_filepath = os.path.join(PATIENT_DATA_PATH, patient_id, f"{patient_id}.txt")

            # .mat file with raw EEG data
            eeg_data_filepath = os.path.join(PATIENT_DATA_PATH, patient_id, f"{patient_id}.txt")

            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            blob_text = bucket.blob(patient_data_filepath)
            # blob_eeg

            # download in memory to reduce I/O operations
            text_file_content  = blob.download_as_text()
            hea_file_content  = blob.download_as_text()
            eeg_file_content = blob.download_as_bytes()


            print(file_content)
            print(type(file_content))


            ### debugging
            # temp = "gs://data-wtwa/i-care-2.0.physionet.org/training/0284/0284.txt"
            # temp = "i-care-2.0.physionet.org/training/0284/0284.txt"
            # data-wtwa/gs://data-wtwa/i-care-2.0.physionet.org/training/0284/0284.txt
            # https://storage.googleapis.com/download/storage/v1/b/data-wtwa/o/gs%3A%2F%2Fdata-wtwa%2Fi-care-2.0.physionet.org%2Ftraining%2F0284%2F0284.txt?alt=media
            ###

        except:
            print("Couldn't download from GCS")
            return

    elif MODEL_TARGET == "local":
        print("importing data from local...")
        # TODO: (low prio) logic to import from local machine
        pass

    else:
        print("env variables not connected")

    print("data imported")


    return
    # return EEG_data, patient_data



# TODO: implement checkpoints to avoid reprocessing in case of failures


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
    import_data("0284")


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
