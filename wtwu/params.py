import os
import numpy as np


##################### VARIABLE ######################

# GCP VARIABLES
GCP_PHYSIONET_BUCKET_NAME = os.environ.get("GCP_PHYSIONET_BUCKET_NAME")

PROJECT_NAME = os.environ.get("PROJECT_NAME")
PROJECT_ID = os.environ.get("PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")

DATASET_ID = os.environ.get("DATASET_ID")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
DATA_TARGET = os.environ.get("DATA_TARGET")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
PATIENT_DATA_PATH = os.environ.get("PATIENT_DATA_PATH")
PATIENT_PROCESSED_DATA_PATH = os.environ.get("PATIENT_PROCESSED_DATA_PATH")

# GCE (Google Compute Engine)
VM_INSTANCE = os.environ.get("VM_INSTANCE")

# BigQuery
BQ_REGION = os.environ.get("BQ_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")

#################### TODO: to eventually copy

# ##################  VARIABLES  ##################
# DATA_SIZE = os.environ.get("DATA_SIZE")
# CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
# MODEL_TARGET = os.environ.get("MODEL_TARGET")
# GCP_PROJECT = os.environ.get("GCP_PROJECT")
# GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
# GCP_REGION = os.environ.get("GCP_REGION")
# BQ_DATASET = os.environ.get("BQ_DATASET")
# BQ_REGION = os.environ.get("BQ_REGION")
# BUCKET_NAME = os.environ.get("BUCKET_NAME")
# INSTANCE = os.environ.get("INSTANCE")
# MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
# MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
# MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
# PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
# PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
# EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
# GAR_IMAGE = os.environ.get("GAR_IMAGE")
# GAR_MEMORY = os.environ.get("GAR_MEMORY")



# ##################  CONSTANTS  #####################
# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
# LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")

# COLUMN_NAMES_RAW = ['fare_amount','pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

# DTYPES_RAW = {
#     "fare_amount": "float32",
#     "pickup_datetime": "datetime64[ns, UTC]",
#     "pickup_longitude": "float32",
#     "pickup_latitude": "float32",
#     "dropoff_longitude": "float32",
#     "dropoff_latitude": "float32",
#     "passenger_count": "int16"
# }

# DTYPES_PROCESSED = np.float32



# ################## VALIDATIONS #################

# env_valid_options = dict(
#     DATA_SIZE=["1k", "200k", "all"],
#     MODEL_TARGET=["local", "gcs", "mlflow"],
# )

# def validate_env_value(env, valid_options):
#     env_value = os.environ[env]
#     if env_value not in valid_options:
#         raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


# for env, valid_options in env_valid_options.items():
#     validate_env_value(env, valid_options)
