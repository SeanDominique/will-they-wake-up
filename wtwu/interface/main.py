from wtwu.packages.storage import *
from wtwu.packages.preprocess import preprocess
from wtwu.packages.model import *

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
    preprocess(patients)
