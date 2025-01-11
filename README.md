# will-they-wake-up project (WTWU)

### EEG Predictive Modeling on Comatose Patients

This repository contains an end-to-end data pipeline and LSTM model for predicting outcomes in over 600 comatose patients based on EEG signals. Our work includes extensive data preprocessing, resampling, and feature engineering, as well as model deployment on Google Cloud Platform (GCP) using virtual machines.

### Overview
	•	Goal: Predict patient outcomes from EEG data using deep learning (LSTM).
	•	Data Volume: 1.5TB raw EEG reduced to ~16GB through optimized preprocessing.
	•	Key Techniques: Resampling, feature extraction and model devlopment inspired from research papers, and reproducible experimentation via modular Python packages.
	•	Result: Achieved a 34% accuracy rate without incorporating ECG data; potential for improvement by integrating additional physiological signals.

### Pipeline & Methodology
	1.	Data Collection: Raw EEG records from 600+ comatose patients - collected from https://moody-challenge.physionet.org/2023/papers/
	2.	Preprocessing on GCP:
	•	Utilized Google Cloud virtual machines to handle large-scale data efficiently.
	•	Converted 1.5TB of raw data to 16GB using downsampling and signal cleaning steps.
	3.	Feature Engineering:
	•	Implemented research-backed techniques.
	4.	LSTM Model:
	•	Architected and trained an LSTM network to classify outcomes.
	•	Packaged modules to enable reproducible training, testing, and hyperparameter tuning.
	5.	Deployment & Orchestration:
	•	Deployed the model on GCP infrastructure.
	•	Streamlined workflows using Python packaging and version control best practices.
