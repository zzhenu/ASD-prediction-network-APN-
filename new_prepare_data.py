#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Data preparation
Usage:
  new_prepare_data.py [--folds=N] [--whole] [--male] [--threshold] [--leave-site-out] [--NYU-site-out][--site=SITE] [--single-site] [<derivative> ...]
  new_prepare_data.py (-h | --help)

Options:
  -h --help           Show this screen
  --folds=N           Number of folds [default: 10]
  --whole             Prepare data of the whole dataset
  --male              Prepare data of male subjects
  --threshold         Prepare data of thresholded subjects
  --leave-site-out    Prepare data using leave-site-out method
  --site=SITE         Prepare data for specific site (e.g., NYU, UM_1)
  --single-site       Prepare data for each site individually
  derivative          Derivatives to process

"""

import numpy as np
import pandas as pd
import os
import random
from docopt import docopt
import numpy.ma as ma
from functools import partial
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils import (load_phenotypes, format_config, run_progress, hdf5_handler)
import re

# Add a function to extract and remove leading zeros from FILE_ID
def extract_id(file_id):
    match = re.search(r'\d+', file_id)
    return int(match.group(0)) if match else None

# Add a function to load subject IDs
def load_subject_ids(file_path):
    with open(file_path, 'r') as f:
        subject_ids = [line.strip() for line in f.readlines()]
    return subject_ids

# Compute the functional connectivity matrix.
def compute_connectivity(functional):
    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(functional))
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        m = ma.masked_where(mask == 1, mask)
        return ma.masked_where(m, corr).compressed()

# Load functional data for a single patient
def load_patient(subj, tmpl):
    df = pd.read_csv(format_config(tmpl, {"subject": subj}), sep="\t", header=0)
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    ROIs = ["#" + str(y) for y in sorted([int(x[1:]) for x in df.keys().tolist()])]
    functional = np.nan_to_num(df[ROIs].to_numpy().T).tolist()
    functional = preprocessing.scale(functional, axis=1)
    functional = compute_connectivity(functional)
    functional = functional.astype(np.float32)
    return subj, functional

# Batch load functional data for multiple patients
def load_patients(subjs, tmpl, jobs=1):
    partial_load_patient = partial(load_patient, tmpl=tmpl)
    msg = "Processing {current} of {total}"
    return dict(run_progress(partial_load_patient, subjs, message=msg, jobs=jobs))

# Prepare the fold data for cross-validation
def prepare_folds(hdf5, folds, pheno, derivatives, experiment):
    exps = hdf5.require_group("experiments")
    ids = pheno["FILE_ID"]
    for derivative in derivatives:
        exp = exps.require_group(format_config(experiment, {"derivative": derivative}))
        exp.attrs["derivative"] = derivative
        skf = StratifiedKFold(n_splits=folds, shuffle=False)
        for i, (train_index, test_index) in enumerate(skf.split(ids, pheno["STRAT"])):
            train_index, valid_index = train_test_split(train_index, test_size=0.111)
            fold = exp.require_group(str(i))
            fold['train'] = [ind.encode('utf8') for ind in ids[train_index]]
            fold['valid'] = [indv.encode('utf8') for indv in ids[valid_index]]
            fold["test"] = [indt.encode('utf8') for indt in ids[test_index]]

# Load patient data into an HDF5 file
def load_patients_to_file(hdf5, pheno, derivatives):
    download_root = "./data/functionals"
    derivatives_path = {
        "aal": "cpac/filt_global/rois_aal/{subject}_rois_aal.1D",
        "cc200": "cpac/filt_global/rois_cc200/{subject}_rois_cc200.1D",
        "dosenbach160": "cpac/filt_global/rois_dosenbach160/{subject}_rois_dosenbach160.1D",
        "ez": "cpac/filt_global/rois_ez/{subject}_rois_ez.1D",
        "ho": "cpac/filt_global/rois_ho/{subject}_rois_ho.1D",
        "tt": "cpac/filt_global/rois_tt/{subject}_rois_tt.1D",
        "cc400": "cpac/filt_global/rois_cc400/{subject}_rois_cc400.1D",
    }
    storage = hdf5.require_group("patients")
    file_ids = pheno["FILE_ID"].tolist()
    for derivative in derivatives:
        file_template = os.path.join(download_root, derivatives_path[derivative])
        func_data = load_patients(file_ids, tmpl=file_template)
        for pid in func_data:
            record = pheno[pheno["FILE_ID"] == pid].iloc[0]
            patient_storage = storage.require_group(pid)
            patient_storage.attrs["id"] = record["FILE_ID"]
            patient_storage.attrs["y"] = record["DX_GROUP"]
            patient_storage.attrs["site"] = record["SITE_ID"]
            patient_storage.attrs["sex"] = record["SEX"]
            patient_storage.attrs["age"] = record["AGE"]
            patient_storage.attrs["site_code"] = record["SITE_ID_CODE"]
            patient_storage.attrs["fiq"] = record["FIQ"]
            patient_storage.create_dataset(derivative, data=func_data[pid])

if __name__ == "__main__":
    random.seed(19)
    np.random.seed(19)

    arguments = docopt(__doc__)

    folds = int(arguments["--folds"])
    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    pheno = load_phenotypes(pheno_path)

    subject_ids_file = "./subject_IDs.txt"
    subject_ids = load_subject_ids(subject_ids_file)

    pheno['ID_extracted'] = pheno['SUB_ID']

    subject_ids = [int(id) for id in subject_ids]

    pheno_filtered = pheno[pheno['ID_extracted'].isin(subject_ids)]


    hdf5 = hdf5_handler(bytes("./data/abide_singe_site_fold5.hdf5", encoding="utf8"), 'a')

    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160","cc400"]
    derivatives = [derivative for derivative in arguments["<derivative>"] if derivative in valid_derivatives]

    load_patients_to_file(hdf5, pheno_filtered, derivatives)

    if "patients" not in hdf5:
        load_patients_to_file(hdf5, pheno_filtered, derivatives)

    if arguments["--whole"]:
        print("Prepare the entire dataset")
        prepare_folds(hdf5, folds, pheno_filtered, derivatives, experiment="{derivative}_whole")

    if arguments["--male"]:
        print("Prepare the male dataset")
        pheno_male = pheno_filtered[pheno_filtered["SEX"] == "M"]
        prepare_folds(hdf5, folds, pheno_male, derivatives, experiment="{derivative}_male")

    if arguments["--threshold"]:
        print("Prepare the threshold-filtered dataset")
        pheno_thresh = pheno_filtered[pheno_filtered["MEAN_FD"] <= 0.2]
        prepare_folds(hdf5, folds, pheno_thresh, derivatives, experiment="{derivative}_threshold")

    if arguments["--leave-site-out"]:
        print("Prepare the leave-site-out dataset")
        for site in pheno_filtered["SITE_ID"].unique():
            if site == 'NYU':
                pheno_without_site = pheno_filtered[pheno_filtered["SITE_ID"] == site]
                prepare_folds(hdf5, folds, pheno_without_site, derivatives, experiment=format_config(
                    "{derivative}_leavesiteout-{site}",
                    {"site": site}
                ))

    if arguments["--NYU-site-out"]:
        print("Prepare the dataset excluding the NYU site")
        pheno_without_site = pheno_filtered[pheno_filtered["SITE_ID"] != 'NYU']
        prepare_folds(hdf5, folds, pheno_without_site, derivatives, experiment="{derivative}_leavesiteout-NYU")

    if arguments["--site"]:
        site = arguments["--site"]
        print(f"Prepare the dataset for site {site}")
        pheno_site = pheno_filtered[pheno_filtered["SITE_ID"] == site]
        prepare_folds(hdf5, folds, pheno_site, derivatives, experiment=format_config(
            "{derivative}_site_{site}",
            {"site": site}
        ))

    if arguments["--single-site"]:
        print("Prepare datasets individually for each site")
        all_sites = pheno_filtered["SITE_ID"].unique()
        for site in all_sites:
            print(f"Processing site: {site}")
            pheno_site = pheno_filtered[pheno_filtered["SITE_ID"] == site]
            prepare_folds(hdf5, folds, pheno_site, derivatives, 
                        experiment=format_config(
                            "{derivative}_singlesite_{site}",
                            {"site": site})
            )
