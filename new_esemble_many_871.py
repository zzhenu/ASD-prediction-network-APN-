#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Autoencoders evaluation.

Usage:
  new_esemble_many_871.py [--whole] [--male] [--threshold] [--leave-site-out] [--NYU-site-out] [--child] [--teen] [--adult] [<derivative> ...]
  new_esemble_many_871.py (-h | --help)

Options:
  -h --help           Show this screen
  --whole             Run model for the whole dataset
  --male              Run model for male subjects
  --threshold         Run model for thresholded subjects
  --leave-site-out    Prepare data using leave-site-out method
  --child             Evaluate on child group (0-12)
  --teen              Evaluate on teen group (13-18)
  --adult             Evaluate on adult group (19-65)
  derivative          Derivatives to process

"""
import numpy as np
import pandas as pd
from docopt import docopt
from new_vae_ae_many_nn import nn
from utils import (load_phenotypes, format_config, hdf5_handler,
                   reset, to_softmax, load_ae_encoder, load_fold,calculate_CDDS_score,load_vae_encoder)
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                            f1_score, roc_auc_score, precision_score, recall_score)
import tensorflow.compat.v1 as tf
from sklearn.feature_selection import SelectKBest, f_classif
import math
import json
import os

def nn_results(hdf5, experiment, code_size_1, code_size_2):
    exp_storage = hdf5["experiments"][experiment]
    n_classes = 2
    fold_results = []
    all_y_true = []
    all_y_prob = []
    
    for fold in exp_storage:
        experiment_cv = format_config("{experiment}_{fold}", {
            "experiment": experiment,
            "fold": fold,
        })

        X_train, y_train, X_valid, y_valid, X_test, y_test = load_fold(
            hdf5["patients"], exp_storage, fold)

        X_all = np.vstack((X_train, X_valid, X_test))
        y_all = np.concatenate((y_train, y_valid, y_test), axis=0)

        X_fc = X_all[:, :-3]
        X_pheno = X_all[:, -3:]

        def get_feature_selection_ratio(exp_name):
            exp_name = exp_name.lower()
            if "aal" in exp_name:
                return 0.18
            elif "ez" in exp_name:
                return 0.35
            elif "cc200" in exp_name:
                return 0.35
            elif exp_name.startswith("ho"):
                return 0.35
            elif exp_name.startswith("dosenbach"):
                return 0.35
            elif exp_name.startswith("cc400"):
                return 0.28
            else:
                return 0.35  # 默认值
        
        ks_ratio = get_feature_selection_ratio(experiment)
        ks = math.ceil(X_fc.shape[1] * ks_ratio)
        cdds_scores = calculate_CDDS_score(X_fc, y_all)

        top_k_indices = np.argsort(cdds_scores)[-ks:]
        X_selected = X_fc[:, top_k_indices]
        

        original_features = X_fc.shape[1]
        selected_features = X_selected.shape[1]
        print(f"Experiment: {experiment}, Fold: {fold}")
        print(f"  Original features: {original_features}")
        print(f"  Selected features: {selected_features} (Top {ks} features, {ks_ratio*100:.1f}% of original)")
        print(f"  Feature reduction: {original_features - selected_features} features removed")

        X_new = np.concatenate((X_selected, X_pheno), axis=1)

        print(f"[Debug] 合并后特征维度: X_new.shape={X_new.shape[1]}")

        train = X_train.shape[0]
        valid = X_valid.shape[0]
        test = X_test.shape[0]
        X_test_mlp = X_new[train+valid:]

        y_test = np.array([to_softmax(n_classes, y) for y in y_test])

        nn_model_path = format_config("./data/best_ckpt/{experiment}_mlp.ckpt", {
            "experiment": experiment_cv,
        })
        
        try:
            model = nn(
                input_size=X_selected.shape[1],
                n_classes=n_classes,
                layers=[
                    {"size": 400, "actv": tf.nn.tanh},
                    {"size": code_size_1, "actv": tf.nn.tanh},
                    {"size": code_size_2, "actv": tf.nn.tanh},
                    {"size": 100, "actv": tf.nn.tanh},
                ],
                pheno_size=3
            )
            
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                saver = tf.train.Saver(model["params"])
                saver.restore(sess, nn_model_path)

                X_test_fc = X_selected[train+valid:]
                X_test_pheno = X_pheno[train+valid:]

                output = sess.run(model["output"], feed_dict={
                    model["input_fc"]: X_test_fc,
                    model["input_pheno"]: X_test_pheno,
                    model["dropouts"][0]: 1.0,
                    model["dropouts"][1]: 1.0,
                    model["dropouts"][2]: 1.0,
                })

                
                # 指标计算
                y_pred = np.argmax(output, axis=1)
                y_true = np.argmax(y_test, axis=1)
                
                [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
                specificity = tn / (fp + tn) if (fp + tn) != 0 else 0
                precision = tp / (tp + fp) if (tp + fp) != 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
                accuracy = accuracy_score(y_true, y_pred)
                fscore = f1_score(y_true, y_pred)
                roc_auc = roc_auc_score(y_true, y_pred)

                fold_results.append([accuracy, precision, fscore, sensitivity, specificity, roc_auc])
                all_y_true.append(y_true)
                all_y_prob.append(output)

        finally:
            reset()

    return {
        "experiment": experiment,
        "fold_metrics": fold_results,
        "all_y_true": all_y_true,
        "all_y_prob": all_y_prob
    }

def calculate_ensemble(experiment_data, fold_idx):
    fold_data = []
    for exp_data in experiment_data:
        fold_metrics = exp_data["fold_metrics"][fold_idx]
        fold_y_true = exp_data["all_y_true"][fold_idx]
        fold_y_prob = exp_data["all_y_prob"][fold_idx]
        
        fold_data.append({
            "experiment": exp_data["experiment"],
            "metrics": fold_metrics,
            "y_true": fold_y_true,
            "y_prob": fold_y_prob
        })

    base_true = fold_data[0]["y_true"]
    for data in fold_data[1:]:
        if not np.array_equal(base_true, data["y_true"]):
            raise ValueError(f"Mismatched true labels in fold {fold_idx+1}")

    accuracies = [data["metrics"][0] for data in fold_data]
    total_acc = sum(accuracies)
    weights = [acc/total_acc for acc in accuracies]

    weighted_probs = np.zeros_like(fold_data[0]["y_prob"])
    for i, data in enumerate(fold_data):
        weighted_probs += weights[i] * data["y_prob"]

    y_pred = np.argmax(weighted_probs, axis=1)
    y_true = fold_data[0]["y_true"]

    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    specificity = tn / (fp + tn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    accuracy = accuracy_score(y_true, y_pred)
    fscore = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    
    return [accuracy, precision, fscore, sensitivity, specificity, roc_auc]

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    reset()
    arguments = docopt(__doc__)
    pd.set_option("display.expand_frame_repr", False)

    # Load data
    pheno = load_phenotypes("./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv")
    hdf5 = hdf5_handler(bytes("./data/abide_871.hdf5", encoding="utf8"), "a")

    # Configure experiments
    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160","cc400"]
    derivatives = [d for d in arguments["<derivative>"] if d in valid_derivatives]
    
    experiments = []
    for derivative in derivatives:
        config = {"derivative": derivative}
        
        # 添加年龄分组实验
        age_groups = []
        if arguments["--child"]:
            age_groups.append("child")
        if arguments["--teen"]:
            age_groups.append("teen")
        if arguments["--adult"]:
            age_groups.append("adult")
            
        for age_group in age_groups:
            experiments.append(format_config(
                "{derivative}_age_{group}", 
                config,
                {"group": age_group}
            ))
        
        if arguments["--whole"]:
            experiments.append(format_config("{derivative}_whole", config))
        if arguments["--male"]:
            experiments.append(format_config("{derivative}_male", config))
        if arguments["--threshold"]:
            experiments.append(format_config("{derivative}_threshold", config))
        if arguments["--leave-site-out"]:
            for site in pheno["SITE_ID"].unique():
                if site == 'NYU':
                    experiments.append(format_config(
                        "{derivative}_leavesiteout-{site}", config, {"site": site}))

    # Run evaluations
    experiment_data = []
    all_fold_results = []
    code_size_1 = 300
    code_size_2 = 200

    for exp in sorted(experiments):
        print(f"\nEvaluating experiment: {exp}")
        result = nn_results(hdf5, exp, code_size_1, code_size_2)
        experiment_data.append(result)

    n_folds = len(experiment_data[0]["fold_metrics"])

    ensemble_fold_results = []
    for fold_idx in range(n_folds):
        fold_metrics = calculate_ensemble(experiment_data, fold_idx)
        ensemble_fold_results.append(fold_metrics)

    ensemble_avg_metrics = np.mean(ensemble_fold_results, axis=0).tolist()

    results = []

    for fold_idx, metrics in enumerate(ensemble_fold_results):
        results.append([f"Fold {fold_idx+1}"] + metrics)

    results.append(["Average"] + ensemble_avg_metrics)

    cols = ["Exp", "Accuracy", "Precision", "F1-score", "Sensitivity", "Specificity", "ROC-AUC"]
    df = pd.DataFrame(results, columns=cols)
    
    print("\n=== Ensemble Results ===")
    print(df)

    import json
    import os
    from datetime import datetime

    metrics_dir = "./evaluation_results/"
    os.makedirs(metrics_dir, exist_ok=True)
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for exp_data in experiment_data:
        output_data = {
            "experiment": exp_data["experiment"],
            "fold_metrics": exp_data["fold_metrics"],
            "all_y_true": [y.tolist() for y in exp_data["all_y_true"]],
            "all_y_prob": [p.tolist() for p in exp_data["all_y_prob"]]
        }

        filename = os.path.join(
            metrics_dir, 
            f"{exp_data['experiment']}_{timestamp}_metrics.json"
        )

        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Evaluation results have been saved: {filename}")

    ensemble_filename = os.path.join(
        metrics_dir,
        f"ensemble_results_{timestamp}.json"
    )
    ensemble_data = {
        "fold_results": ensemble_fold_results,
        "average_metrics": ensemble_avg_metrics
    }
    with open(ensemble_filename, 'w') as f:
        json.dump(ensemble_data, f, indent=2)
    print(f"Ensemble learning results have been saved: {ensemble_filename}")