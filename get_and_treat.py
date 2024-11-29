import wtwu.packages.storage as wtstorage
import wtwu.packages.data as wtdata
import os
import numpy as np
import pickle
from google.cloud import storage


# Paramètres du bucket
bucket_name = "data-wtwa"
preprocessed_path = "preprocessed/"  # Dossier cible dans le bucket Google Cloud
local_processed_path = "./data/processed"  # Dossier temporaire local

# Initialisation du client Google Cloud Storage
client = storage.Client()

def get_list_of_patients(bucket_name, training_path):
    """
    Récupère la liste des patients à partir des blobs présents dans le bucket Google Storage.
    """
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=training_path)

    # Extraire les ID des patients
    patient_ids = set()
    for blob in blobs:
        path_parts = blob.name.split("/")
        if len(path_parts) > 2 and path_parts[-2].isnumeric():
            patient_ids.add(path_parts[-2])

    return sorted(list(patient_ids))

def preprocess_patient(patient, local_path):
    """
    Prétraite les données d'un patient et sauvegarde les fichiers localement.
    """
    patient_local_path = os.path.join(local_path, str(patient))
    if not os.path.exists(f'{patient_local_path}/times.npy'):

        # Import des données
        survived, eeg_data_headers, all_eeg_data = wtstorage.import_data(patient)

        if eeg_data_headers != 'Error' and len(eeg_data_headers) > 0:
            fs = eeg_data_headers[0]['fs']
            hours = np.array([header['recording_hour'] for header in eeg_data_headers]).astype(np.float16)

            # Création du dossier local pour le patient
            os.makedirs(patient_local_path, exist_ok=True)

            # Écriture des métadonnées
            with open(f'{patient_local_path}/y.txt', 'a+') as f:
                f.write(f'survived:{survived}\n')

            # Réduction des données
            reduced_eeg_data = wtdata.reduce_all_channels(all_eeg_data, target_freq=100, original_freq=fs)

            # Fenêtrage et normalisation
            list_of_splits, list_of_times = wtdata.sample_all(reduced_eeg_data, hours=hours)
            std = np.std(list_of_splits, axis=0)
            mean = np.mean(list_of_splits, axis=0)
            std = np.where(std == 0, 1, std)
            list_of_splits = (list_of_splits - mean) / std

            # Calcul des PSD
            psds_fs, list_of_psds = wtdata.get_psds(list_of_splits)

            # Sauvegarde des fichiers localement
            np.save(f'{patient_local_path}/psds.npy', list_of_psds)
            np.save(f'{patient_local_path}/psds_fs.npy', psds_fs)
            np.save(f'{patient_local_path}/time_splits.npy', list_of_splits)
            pickle.dump(eeg_data_headers, open(f'{patient_local_path}/headers.pkl', 'wb'))
            np.save(f'{patient_local_path}/times.npy', list_of_times)

            print(f"Patient {patient} : Prétraitement terminé!")
            return patient_local_path
        else:
            print(f"Patient {patient} : Données introuvables ou incorrectes.")
    else:
        print(f"Patient {patient} : Données déjà prétraitées.")
    return None

def upload_preprocessed_to_gcp(patient_local_path, bucket_name, preprocessed_path, patient_id):
    """
    Upload les fichiers prétraités pour un patient vers le bucket GCP.
    """
    bucket = client.bucket(bucket_name)
    remote_path = f"{preprocessed_path}{patient_id}/"

    for file_name in os.listdir(patient_local_path):
        local_file_path = os.path.join(patient_local_path, file_name)
        blob_path = f"{remote_path}{file_name}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_file_path)
        print(f"Fichier {file_name} uploadé vers {blob_path}")

def delete_local_files(patient_local_path):
    """
    Supprime les fichiers locaux d'un patient après leur upload.
    """
    if os.path.exists(patient_local_path):
        for file_name in os.listdir(patient_local_path):
            file_path = os.path.join(patient_local_path, file_name)
            os.remove(file_path)  # Supprimer chaque fichier
        os.rmdir(patient_local_path)  # Supprimer le dossier une fois vide
        print(f"Fichiers locaux pour le patient supprimés : {patient_local_path}")

def preprocess_and_upload(bucket_name, training_path, local_processed_path, preprocessed_path):
    """
    Prétraite les données de tous les patients, les upload vers le bucket Google Cloud,
    puis supprime les fichiers locaux.
    """
    patients = get_list_of_patients(bucket_name, training_path)
    print(f"Patients détectés : {patients}")

    for patient in patients:
        print(f"Traitement du patient {patient}...")
        patient_local_path = preprocess_patient(patient, local_processed_path)
        if patient_local_path:
            upload_preprocessed_to_gcp(patient_local_path, bucket_name, preprocessed_path, patient)
            delete_local_files(patient_local_path)

# Lancer le processus
preprocess_and_upload(bucket_name, "i-care-2.0.physionet.org/training/", local_processed_path, preprocessed_path)
