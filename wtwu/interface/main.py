from wtwu.packages.storage import *
from wtwu.packages.preprocess import preprocess
from wtwu.packages.models import *
from wtwu.params import *

import numpy as np
from google.cloud import storage



def preproc(patients: list):
    """
    Patients can be an empty list. If the case, will fetch the entire patient list from GCS.
    """
    preprocess(patients)
    return


def train(input_shape, model, ):
    # model
    # create_model()
    pass


def pred():
    # give the address to the new data OR the new data
    pass


if __name__ == "__main__":

    patients = []

    # check latest processed patient data
    try:
        client = storage.Client()
        print("check 0")
        blobs = client.list_blobs(BUCKET_NAME, prefix=f"{PATIENT_PROCESSED_DATA_PATH}")

        blob_names = set()

        for blob in blobs:
            patient_id = blob.name.split("/")[-2]
            blob_names.add(patient_id)

        blob_names = sorted(blob_names)
        print(blob_names)
        last_processed_patient_id = blob_names[-1]

        # continue preprocessing from where it left off
        with open("patient_ids.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                patient_id = line.split("/")[-2]

                if int(patient_id) > int(last_processed_patient_id):
                    patients.append(patient_id)


    except Exception as e:
        print(f"error finding preprocessed patients in GCS bucket : {e}")

    print("patient to preprocess : \n", patients)

    preprocess(patients)
