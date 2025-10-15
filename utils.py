#!/usr/bin/env python
import os
import re
import sys
import h5py
import time
import string
import contextlib
import multiprocessing
import pandas as pd
import numpy as np
import tensorflow as tf
from model import ae,vae
from tensorflow.python.framework import ops
from sklearn.preprocessing import MinMaxScaler
import tensorflow.compat.v1 as tf

identifier = '(([a-zA-Z]_)?([a-zA-Z0-9_]*))'
replacement_field = '{' + identifier + '}'

#CDDS
def calculate_CDDS_score(X, y, n_bins=20):
    print("======================Start CDDS score calculation=====================")
    """
    X: feature matrix (n_samples, n_features)
    y: label vector (n_samples,)
    n_bins: number of bins
    Returns the DSDC score for each feature
    """
    #Get the number of samples and features
    n_samples, n_features = X.shape
    #Initialize storage for CDDS groups of each feature
    scores = np.zeros(n_features)
    
    # Generate boolean masks for positive class (e.g., ASD) and negative class (e.g., HC)
    pos_mask = (y == 1)# Positive class (samples with label 1)
    neg_mask = (y == 0)# Negative class (samples with label 0)
    
    # Compute the number of samples in the positive and negative classes
    n_pos = np.sum(pos_mask)# Total number of positive class samples
    n_neg = np.sum(neg_mask)# Total number of negative class samples
    
    # Iterate over each feature and compute its CDDS score
    for i in range(n_features):
        # Extract all sample values for the current feature
        feature = X[:, i]
        
        # Compute the minimum and maximum values of the current feature
        min_val, max_val = np.min(feature), np.max(feature)
        
        # Evenly divide the feature value range into n_bins + 1 bin edges
        bin_edges = np.linspace(min_val, max_val, n_bins+1)
        
        # Count the number of positive class samples in each bin
        pos_counts = np.histogram(feature[pos_mask], bins=bin_edges)[0]
        # Count the number of negative class samples in each bin
        neg_counts = np.histogram(feature[neg_mask], bins=bin_edges)[0]
        
        # Compute the probability distribution of positive and negative class samples in each bin (normalize)
        pos_probs = pos_counts / n_pos
        neg_probs = neg_counts / n_neg
        
        #Compute the cumulative sum of the differences between positive and negative class probabilities
        cumulative_diff = np.cumsum(pos_probs - neg_probs)
        
        # Sum of the absolute values of the cumulative differences for CDDS
        scores[i] = np.sum(np.abs(cumulative_diff))
        
    # Return the CDDS scores for all features
    return scores

def reset():
    ops.reset_default_graph()
    np.random.seed(19)
    tf.set_random_seed(19)

    import random
    random.seed(19)

    from sklearn.utils import check_random_state
    check_random_state(19)

def load_phenotypes(pheno_path):
    pheno = pd.read_csv(pheno_path)

    pheno['DX_GROUP'] = pheno['DX_GROUP'].apply(lambda v: int(v)-1)
    pheno['SITE_ID'] = pheno['SITE_ID'].apply(lambda v: re.sub('_[0-9]', '', v))

    pheno["FIQ"] = pheno['FIQ'].fillna(pheno['FIQ'].mean())

    pheno['SITE_ID_CODE'] = pd.factorize(pheno['SITE_ID'])[0].astype(int)

    pheno['SEX'] = pheno['SEX']
    pheno['MEAN_FD'] = pheno['func_mean_fd']
    pheno['SUB_IN_SMP'] = pheno['SUB_IN_SMP'].apply(lambda v: v == 1)
    pheno["STRAT"] = pheno[["SITE_ID", "DX_GROUP"]].apply(lambda x: "_".join([str(s) for s in x]), axis=1)
    pheno["AGE"] = pheno['AGE_AT_SCAN']

    pheno.index = pheno['FILE_ID']

    return pheno[['SUB_ID','FILE_ID', 'DX_GROUP', 'SEX', 'SITE_ID', 'SITE_ID_CODE','MEAN_FD', 'SUB_IN_SMP', 'STRAT','AGE', 'FIQ']]


def load_phenotypes_2(pheno_path):

    pheno = pd.read_csv(pheno_path)
    
    pheno = pheno[pheno['FILE_ID'] != 'no_filename']
    pheno['DX_GROUP'] = pheno['DX_GROUP'].apply(lambda v: int(v)                                  -1)
    pheno['SITE_ID'] = pheno['SITE_ID'].apply(lambda v: re.sub('_[0-9]', '', v))
    pheno['SEX'] = pheno['SEX'].apply(lambda v: {1: "M", 2: "F"}[v])
    pheno['SEX'] = pheno['SEX']
    pheno['MEAN_FD'] = pheno['func_mean_fd']
    pheno['SUB_IN_SMP'] = pheno['SUB_IN_SMP'].apply(lambda v: v == 1)
    pheno["STRAT"] = pheno[["SITE_ID", "DX_GROUP"]].apply(lambda x: "_".join([str(s) for s in x]), axis=1)
    pheno["AGE"] = pheno['AGE_AT_SCAN']
    pheno["FIQ"] = pheno['FIQ'].fillna(pheno['FIQ'].mean())

    pheno['HANDEDNESS_SCORES'] = pheno['HANDEDNESS_SCORES'].fillna(method='bfill')
    pheno.index = pheno['FILE_ID']

    return pheno[['FILE_ID', 'DX_GROUP', 'SEX', 'SITE_ID', 'MEAN_FD', 'SUB_IN_SMP', 'STRAT','AGE','HANDEDNESS_SCORES','FIQ']]


def hdf5_handler(filename, mode="r"):
    h5py.File(filename, "a").close()
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[1] = 0
    settings[2] = 0
    propfaid.set_cache(*settings)
    with contextlib.closing(h5py.h5f.open(filename, fapl=propfaid)) as fid:
        f = h5py.File(fid, mode)
        return f

#Load the training, validation, and test data for the specified fold from the HDF5 file, and append phenotypic information (SITEID, Age, FIQ) to the feature vectors
def load_fold(patients, experiment, fold):
    derivative = experiment.attrs["derivative"]
    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    pheno = load_phenotypes(pheno_path)

    X_train = []
    y_train = []
    X_valid = []
    y_valid = []
    X_test = []
    y_test = []

    train_fiq = []

    for pid in experiment[fold]["train"]:
        p = pheno[pheno['FILE_ID'] == pid.decode(encoding='UTF-8')]
        train_fiq.append(float(p['FIQ'].values[0]))

    from sklearn.preprocessing import StandardScaler
    fiq_scaler = StandardScaler()
    fiq_scaler.fit(np.array(train_fiq).reshape(-1, 1))

    for pid in experiment[fold]["train"]:
        p = pheno[pheno['FILE_ID'] == pid.decode(encoding='UTF-8')]
        x = np.array(patients[pid][derivative])

        x = np.append(x, float(p['AGE'].values[0]))
        x = np.append(x, int(p['SITE_ID_CODE'].values[0]))

        fiq_val = fiq_scaler.transform([[float(p['FIQ'].values[0])]])[0][0]
        x = np.append(x, fiq_val)
        
        X_train.append(x)
        y_train.append(patients[pid].attrs["y"])

    for pid in experiment[fold]["valid"]:
        p = pheno[pheno['FILE_ID'] == pid.decode(encoding='UTF-8')]
        x = np.array(patients[pid][derivative])

        x = np.append(x, float(p['AGE'].values[0]))#1
        x = np.append(x, int(p['SITE_ID_CODE'].values[0]))#2

        fiq_val = fiq_scaler.transform([[float(p['FIQ'].values[0])]])[0][0]
        x = np.append(x, fiq_val)#3
        
        X_valid.append(x)
        y_valid.append(patients[pid].attrs["y"])

    for pid in experiment[fold]["test"]:
        p = pheno[pheno['FILE_ID'] == pid.decode(encoding='UTF-8')]
        x = np.array(patients[pid][derivative])

        x = np.append(x, float(p['AGE'].values[0]))#1
        x = np.append(x, int(p['SITE_ID_CODE'].values[0]))#2

        fiq_val = fiq_scaler.transform([[float(p['FIQ'].values[0])]])[0][0]
        x = np.append(x, fiq_val)#3
        
        X_test.append(x)
        y_test.append(patients[pid].attrs["y"])

    return np.array(X_train), y_train, \
           np.array(X_valid), y_valid, \
           np.array(X_test), y_test


class SafeFormat(dict):

    def __missing__(self, key):
        return "{" + key + "}"

    def __getitem__(self, key):
        if key not in self:
            return self.__missing__(key)
        return dict.__getitem__(self, key)


def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def format_config(s, *d):
    dd = merge_dicts(*d)
    return string.Formatter().vformat(s, [], SafeFormat(dd))


def elapsed_time(tstart):
    tnow = time.time()
    total = tnow - tstart
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def run_progress(callable_func, items, message=None, jobs=1):

    results = []

    print ('Starting pool of %d jobs' % jobs)

    current = 0
    total = len(items)

    if jobs == 1:
        results = []
        for item in items:
            results.append(callable_func(item))
            current = len(results)
            if message is not None:
                args = {'current': current, 'total': total}
                sys.stdout.write("\r" + message.format(**args))
                sys.stdout.flush()

    # Or allocate a pool for multithreading
    else:
        pool = multiprocessing.Pool(processes=jobs)
        for item in items:
            pool.apply_async(callable_func, args=(item,), callback=results.append)

        while current < total:
            current = len(results)
            if message is not None:
                args = {'current': current, 'total': total}
                sys.stdout.write("\r" + message.format(**args))
                sys.stdout.flush()
            time.sleep(0.5)

        pool.close()
        pool.join()

    return results


def root():
    return os.path.dirname(os.path.realpath(__file__))


def to_softmax(n_classes, classe):
    sm = [0.0] * n_classes
    sm[int(classe)] = 1.0
    return sm


def load_ae_encoder(input_size, code_size, model_path):
    tf.compat.v1.disable_eager_execution()  # 新增此行
    model = ae(input_size, code_size)
    init = tf.global_variables_initializer()
    try:
        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver(model["params"], write_version= tf.train.SaverDef.V2)
            if os.path.isfile(model_path):
                print ("Restoring", model_path)
                saver.restore(sess, model_path)
            params = sess.run(model["params"])
            return {"W_enc": params["W_enc"], "b_enc": params["b_enc"]}
    finally:
        reset()
        
def load_vae_encoder(input_size, latent_dim, model_path):
    tf.compat.v1.disable_eager_execution()
    kl_weight_ph = tf.placeholder(tf.float32)

    model = vae(input_size, latent_dim, kl_weight_ph)

    try:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 初始化所有变量
            saver = tf.train.Saver()
            if os.path.isfile(model_path):
                print("Restoring VAE from", model_path)
                saver.restore(sess, model_path)

            encoder_params = {
                "encoder_h1/kernel": sess.run("encoder/encoder_h1/kernel:0"),
                "encoder_h1/bias": sess.run("encoder/encoder_h1/bias:0"),
                # Shared layer mu_logvar_layer (generates μ and logvar)
                "mu_logvar_layer/kernel": sess.run("encoder/mu_logvar_layer/kernel:0"),
                "mu_logvar_layer/bias": sess.run("encoder/mu_logvar_layer/bias:0")
            }
            return encoder_params
    finally:
        reset()

def sparsity_penalty(x, p, coeff):
    p_hat = tf.reduce_mean(tf.abs(x), 0)
    kl = p * tf.log(p / p_hat) + \
        (1 - p) * tf.log((1 - p) / (1 - p_hat))
    return coeff * tf.reduce_sum(kl)
