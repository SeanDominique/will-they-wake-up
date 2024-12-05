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
        blobs = client.list_blobs({BUCKET_NAME}, prefix={PATIENT_PROCESSED_DATA_PATH}, delimiter="/")

        blob_names = set()
        for blob in blobs:
            blob_names.add(blob.name)

        blob_names = sorted(blob_names)
        last_processed_patient_id = blob_names[-1]


    except:
        print("error finding preprocessed patients in GCS bucket")

    # continue preprocessing from where it left off
    with open("patient_ids.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            patient_id = line.split("/")[-2]

            if int(patient_id) > int(last_processed_patient_id):
                patients.append(patient_id)

    print(patients)

    # preprocess(patients)
