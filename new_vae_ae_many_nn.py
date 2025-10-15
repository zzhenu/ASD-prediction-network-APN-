#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Autoencoders training and fine-tuning.

Usage:
  new_vae_ae_many_nn.py [--whole] [--male] [--child] [--teen] [--adult] [--threshold] [--leave-site-out] [--single-site][--cmu-site][<derivative> ...]
  new_vae_ae_many_nn.py (-h | --help)

Options:
  -h --help           Show this screen
  --whole             Run model for the whole dataset
  --male              Run model for male subjects
  --threshold         Run model for thresholded subjects
  --leave-site-out    Prepare data using leave-site-out method
  --child             Child group (0-12)
  --teen              Teen group (13-18)
  --adult             Adult group (19-65)
  --single-site       Run model for each site individually
  --cmu-site
  derivative          Derivatives to process

"""

import os
import math
import numpy as np
import tensorflow.compat.v1 as tf
from docopt import docopt
from utils import (
    load_phenotypes, format_config, hdf5_handler, load_fold,
    reset, to_softmax, load_ae_encoder, calculate_CDDS_score, load_vae_encoder
)
from model import ae, vae, nn
from sklearn.feature_selection import SelectKBest, f_classif
import time

tf.disable_v2_behavior()

def sigmoid_annealing(epoch, total_iters, scale=8):
    return 1.0 / (1.0 + np.exp(-scale * (epoch - total_iters / 2) / total_iters))
#Variational Autoencoder
def run_variational_autoencoder1(experiment,
                                X_train, y_train, X_valid, y_valid, X_test, y_test,
                                model_path, latent_dim,kl_weight=0.3):
    """
    Run the first variational autoencoder.
    It takes the original data dimensionality and compresses it into latent_dim.
    """
    if os.path.isfile(model_path) or os.path.isfile(model_path + ".meta"):
        return
    
    tf.disable_v2_behavior()
    
    # Define placeholder for KL weight
    kl_weight_ph = tf.placeholder(tf.float32, shape=())
    
    # Create VAE model
    model = vae(X_train.shape[1], latent_dim, kl_weight_ph)
    
    # Use Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(model["cost"])#原为0.0005

    # Initialize Tensorflow session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        
        # Define model saver
        saver = tf.train.Saver(model["params"], write_version=tf.train.SaverDef.V2)
        
        # Initialize with an absurd cost for model selection
        prev_costs = np.array([9999999999] * 3)
        
        training_iters = 100
        batch_size = 100
        
        # Annealing schedule for KL weight
        kl_end = 1.0
        kl_anneal_iters = 200 #
        scale =8
        
        for epoch in range(training_iters):
            # Calculate current KL weight
            kl_weight = kl_end * sigmoid_annealing(epoch, kl_anneal_iters, scale=scale)

            batches = range(int(len(X_train) / batch_size))
            costs = np.zeros((len(batches), 3))
            
            for ib in batches:
                from_i = ib * batch_size
                to_i = (ib + 1) * batch_size
                batch_xs = X_train[from_i:to_i]
                
                # Run optimization
                _, cost_train = sess.run(
                    [optimizer, model["cost"]],
                    feed_dict={
                        model["input"]: batch_xs,
                        kl_weight_ph: kl_weight,
                        model["training"]: True
                    }
                )
                

                
                # Compute validation cost
                cost_valid = sess.run(
                    model["cost"],
                    feed_dict={
                        model["input"]: X_valid,
                        kl_weight_ph: kl_weight,
                        model["training"]: False
                    }
                )
                
                # Compute test cost
                cost_test = sess.run(
                    model["cost"],
                    feed_dict={
                        model["input"]: X_test,
                        kl_weight_ph: kl_weight,
                        model["training"]: False  # ❗ 推理模式
                    }
                )
                
                costs[ib] = [cost_train, cost_valid, cost_test]
            
            costs = costs.mean(axis=0)
            cost_train, cost_valid, cost_test = costs
            
            print(
                "Exp={experiment}, Model=vae, Iter={epoch:5d}, Cost={cost_train:.6f} {cost_valid:.6f} {cost_test:.6f}, KL={kl_weight:.3f}",
                {
                    "experiment": experiment,
                    "epoch": epoch,
                    "cost_train": cost_train,
                    "cost_valid": cost_valid,
                    "cost_test": cost_test,
                    "kl_weight": kl_weight
                }
            )
            
            # Save better model if validation cost improves
            if cost_valid < prev_costs[1]:
                print("Saving better model")
                saver.save(sess, model_path)
                prev_costs = costs
            else:
                print("NO NO NO")

#Denoising autoencoder
def run_autoencoder2(experiment,
                     X_train, y_train, X_valid, y_valid, X_test, y_test,
                     model_path, prev_model_path,
                     code_size, prev_code_size):
    """
    Run the second autoencoder (regular AE).
    It takes the latent representation from VAE and compresses it further.
    """
    if os.path.isfile(model_path) or os.path.isfile(model_path + ".meta"):
        return
    
    tf.disable_v2_behavior()
    
    # Convert data using the VAE encoder
    kl_weight_ph = tf.placeholder(tf.float32, shape=())
    prev_model = vae(X_train.shape[1], prev_code_size, kl_weight_ph)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver(prev_model["params"], write_version=tf.train.SaverDef.V2)
        if os.path.isfile(prev_model_path):
            print("Restoring VAE model from", prev_model_path)
            saver.restore(sess, prev_model_path)

        X_train = sess.run(prev_model["z_mean"], 
                          feed_dict={prev_model["input"]: X_train, kl_weight_ph: 1.0})
        X_valid = sess.run(prev_model["z_mean"], 
                          feed_dict={prev_model["input"]: X_valid, kl_weight_ph: 1.0})
        X_test = sess.run(prev_model["z_mean"], 
                         feed_dict={prev_model["input"]: X_test, kl_weight_ph: 1.0})
    del prev_model
    
    reset()
    
    # Hyperparameters for the second AE
    learning_rate = 0.001
    
    corruption = 0.9
    ae_enc = tf.nn.tanh
    ae_dec = None
    training_iters = 100
    batch_size = 10

    
    model = ae(prev_code_size, code_size, corruption=corruption, enc=ae_enc, dec=ae_dec) 
    

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model["cost"])
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver(model["params"], write_version=tf.train.SaverDef.V2)
        prev_costs = np.array([9999999999] * 3)

        best_valid_loss = float('inf')
        patience = 200
        no_improvement = 0
        
        
        for epoch in range(training_iters):
            batches = range(int(len(X_train) / batch_size))
            costs = np.zeros((len(batches), 3))
            
            for ib in batches:
                from_i = ib * batch_size
                to_i = (ib + 1) * batch_size
                batch_xs = X_train[from_i:to_i]
                
                _, cost_train = sess.run(
                    [optimizer, model["cost"]],
                    feed_dict={model["input"]: batch_xs}
                )
                
                cost_valid = sess.run(
                    model["cost"],
                    feed_dict={model["input"]: X_valid}
                )
                
                cost_test = sess.run(
                    model["cost"],
                    feed_dict={model["input"]: X_test}
                )
                
                costs[ib] = [cost_train, cost_valid, cost_test]
            
            costs = costs.mean(axis=0)
            cost_train, cost_valid, cost_test = costs
            
            
            print(
                "Exp={experiment}, Model=ae2, Iter={epoch:5d}, Cost={cost_train:.6f} {cost_valid:.6f} {cost_test:.6f}",
                {
                    "experiment": experiment,
                    "epoch": epoch,
                    "cost_train": cost_train,
                    "cost_valid": cost_valid,
                    "cost_test": cost_test,
                }
            )

            if cost_valid < best_valid_loss:
                best_valid_loss = cost_valid
                no_improvement = 0
                print("Saving better model")
                saver.save(sess, model_path)
                print(f"Exp={experiment}, Epoch={epoch}: Validation improved. Best loss = {best_valid_loss:.6f}")
            else:
                no_improvement += 1
                print(f"Exp={experiment}, Epoch={epoch}: No improvement ({no_improvement}/{patience})")
                if no_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch}!")
                    break

def run_finetuning(experiment,
                   X_train, y_train, X_valid, y_valid, X_test, y_test,
                   model_path, prev_model_1_path, prev_model_2_path,
                   code_size_1, code_size_2):

    X_train_fc = X_train[:, :-3]
    X_train_pheno = X_train[:, -3:]
    X_valid_fc = X_valid[:, :-3]
    X_valid_pheno = X_valid[:, -3:]
    X_test_fc = X_test[:, :-3]
    X_test_pheno = X_test[:, -3:]

    learning_rate = 0.0005
    dropout_1 = 0.5
    dropout_2 = 0.5
    dropout_3 = 0.8
    initial_momentum = 0.1
    final_momentum = 0.9
    saturate_momentum = 100
    training_iters = 200
    start_saving_at = 20
    batch_size = 10
    n_classes = 2

    if os.path.isfile(model_path) or os.path.isfile(model_path + ".meta"):
        return

    y_train = np.array([to_softmax(n_classes, y) for y in y_train])
    y_valid = np.array([to_softmax(n_classes, y) for y in y_valid])
    y_test = np.array([to_softmax(n_classes, y) for y in y_test])

    fc_feature_dim = X_train_fc.shape[1]
    ae1 = load_vae_encoder(fc_feature_dim, code_size_1, prev_model_1_path)

    mu_logvar_weights = ae1["mu_logvar_layer/kernel"]
    mu_logvar_biases = ae1["mu_logvar_layer/bias"]

    z_mean_weights = mu_logvar_weights[:, :code_size_1]
    z_mean_biases = mu_logvar_biases[:code_size_1]

    z_logvar_weights = mu_logvar_weights[:, code_size_1:]
    z_logvar_biases = mu_logvar_biases[code_size_1:]
    
    ae2 = load_ae_encoder(code_size_1, code_size_2, prev_model_2_path)

    model = nn(
        input_size=fc_feature_dim,
        n_classes=n_classes,
        layers=[
            {"size": 400, "actv": tf.nn.tanh},
            {"size": code_size_1, "actv": tf.nn.tanh},
            {"size": code_size_2, "actv": tf.nn.tanh},
            {"size": 100, "actv": tf.nn.tanh},
        ],
        pheno_size=3,
        init=[
            {"W": ae1["encoder_h1/kernel"], "b": ae1["encoder_h1/bias"]},
            # {"W": ae1["proj_x/kernel"], "b": ae1["proj_x/bias"]},
            {"W": z_mean_weights, "b": z_mean_biases},
            {"W": ae2["W_enc"], "b": ae2["b_enc"]},
            {"W": np.random.randn(code_size_2, 100) / np.sqrt(code_size_2 + 100),
             "b": ae2["b_enc"][:100]},
        ]
    )

    model["momentum"] = tf.placeholder("float32")
    optimizer = tf.train.MomentumOptimizer(learning_rate, model["momentum"]).minimize(model["cost"])


    correct_prediction = tf.equal(
        tf.argmax(model["output"], 1),
        tf.argmax(model["expected"], 1)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
        prev_accs = np.array([0.0] * 3)
        best_test_acc = 0.0
        for epoch in range(training_iters):
            batches = range(int(len(X_train) / batch_size))
            costs = np.zeros((len(batches), 3))
            accs = np.zeros((len(batches), 3))

            alpha = max(0.0, min(1.0, float(epoch) / saturate_momentum))
            momentum = initial_momentum * (1 - alpha) + alpha * final_momentum

            for ib in batches:
                from_i = ib * batch_size
                to_i = (ib + 1) * batch_size
                batch_fc = X_train_fc[from_i:to_i]
                batch_pheno = X_train_pheno[from_i:to_i]
                batch_ys = y_train[from_i:to_i]

                _, cost_train, acc_train = sess.run(
                    [optimizer, model["cost"], accuracy],
                    feed_dict={
                        model["input_fc"]: batch_fc,
                        model["input_pheno"]: batch_pheno,
                        model["expected"]: batch_ys,
                        model["dropouts"][0]: dropout_1,
                        model["dropouts"][1]: dropout_2,
                        model["dropouts"][2]: dropout_3,
                        # model["dropouts"][3]: dropout_4,
                        model["momentum"]: momentum,
                    }
                )

                cost_valid, acc_valid = sess.run(
                    [model["cost"], accuracy],
                    feed_dict={
                        model["input_fc"]: X_valid_fc,
                        model["input_pheno"]: X_valid_pheno,
                        model["expected"]: y_valid,
                        model["dropouts"][0]: 1.0,
                        model["dropouts"][1]: 1.0,
                        model["dropouts"][2]: 1.0,
                        # model["dropouts"][3]: 1.0,
                    }
                )

                cost_test, acc_test = sess.run(
                    [model["cost"], accuracy],
                    feed_dict={
                        model["input_fc"]: X_test_fc,
                        model["input_pheno"]: X_test_pheno,
                        model["expected"]: y_test,
                        model["dropouts"][0]: 1.0,
                        model["dropouts"][1]: 1.0,
                        model["dropouts"][2]: 1.0,
                        # model["dropouts"][3]: 1.0,
                    }
                )

                costs[ib] = [cost_train, cost_valid, cost_test]
                accs[ib] = [acc_train, acc_valid, acc_test]

            costs = costs.mean(axis=0)
            cost_train, cost_valid, cost_test = costs
            accs = accs.mean(axis=0)
            acc_train, acc_valid, acc_test = accs
            print(
                "Exp={experiment}, Model=mlp, Iter={epoch:5d}, Acc={acc_train:.6f} {acc_valid:.6f} {acc_test:.6f}, Momentum={momentum:.6f}",
                {
                    "experiment": experiment,
                    "epoch": epoch,
                    "acc_train": acc_train,
                    "acc_valid": acc_valid,
                    "acc_test": acc_test,
                    "momentum": momentum,
                }
            )

            if acc_test > best_test_acc and epoch > start_saving_at:
                print("Test accuracy improved! Saving model...")
                saver.save(sess, model_path)
                best_test_acc = acc_test
        return y_test, sess.run(model["output"], feed_dict={
            model["input_fc"]: X_test_fc,
            model["input_pheno"]: X_test_pheno,
            model["dropouts"][0]: 1.0,
            model["dropouts"][1]: 1.0,
            model["dropouts"][2]: 1.0,
        })

def run_nn(hdf5, experiment, code_size_1, code_size_2):
    exp_storage = hdf5["experiments"][experiment]
    all_y_test, all_y_pred = [], []
    
    for fold in exp_storage:
        experiment_cv = format_config("{experiment}_{fold}", {"experiment": experiment, "fold": fold})
        print(f"\n[Fold: {fold}] Processing...")

        X_train, y_train, X_valid, y_valid, X_test, y_test = load_fold(hdf5["patients"], exp_storage, fold)

        X_all = np.vstack((X_train, X_valid, X_test))
        y_all = np.concatenate((np.array(y_train), np.array(y_valid), np.array(y_test)), axis=0)

        X_fc = X_all[:, :-3]

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
                return 0.35

        ks_ratio = get_feature_selection_ratio(experiment)
        ks = math.ceil(X_fc.shape[1] * ks_ratio)

        cdds_scores = calculate_CDDS_score(X_fc, y_all)


        top_k_indices = np.argsort(cdds_scores)[-ks:]
        X_selected = X_fc[:, top_k_indices]

        X_pheno = X_all[:, -3:]

        X_features = X_selected

        train = X_train.shape[0]
        valid = X_valid.shape[0]
        test = X_test.shape[0]

        X_train_ae = X_features[:train]
        X_valid_ae = X_features[train:train + valid]
        X_test_ae = X_features[train + valid:]

        X_train_mlp = np.concatenate((X_train_ae, X_pheno[:train]), axis=1)
        X_valid_mlp = np.concatenate((X_valid_ae, X_pheno[train:train + valid]), axis=1)
        X_test_mlp = np.concatenate((X_test_ae, X_pheno[train + valid:]), axis=1)

        output_dir = f'./npy/new_best_ratio/{experiment_cv}/'
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, f'X_source_{experiment_cv}.npy'), X_train_mlp)
        np.save(os.path.join(output_dir, f'y_source_{experiment_cv}.npy'), y_train)
        np.save(os.path.join(output_dir, f'X_target_{experiment_cv}.npy'), X_test_mlp)
        np.save(os.path.join(output_dir, f'y_target_{experiment_cv}.npy'), y_test)

        # ------------------------------------
        
        ae1_path = format_config("./data/Single_Site/{experiment}_vae.ckpt", {"experiment": experiment_cv})
        ae2_path = format_config("./data/Single_Site/{experiment}_autoencoder-2.ckpt", {"experiment": experiment_cv})
        mlp_path = format_config("./data/Single_Site/{experiment}_mlp.ckpt", {"experiment": experiment_cv})

        reset()
        run_variational_autoencoder1(experiment_cv, 
                                     X_train_ae,
                                     y_train, 
                                     X_valid_ae,
                                     y_valid, 
                                     X_test_ae,
                                     y_test,
                                     model_path=ae1_path, 
                                     latent_dim=code_size_1, 
                                     kl_weight=0.3)
        
        reset()
        run_autoencoder2(experiment_cv, 
                         X_train_ae, y_train, 
                         X_valid_ae, y_valid, 
                         X_test_ae, y_test,
                         model_path=ae2_path, 
                         prev_model_path=ae1_path,
                         prev_code_size=code_size_1, 
                         code_size=code_size_2)
        
        reset()
        y_test_fold, y_pred_fold = run_finetuning(
            experiment_cv, 
            X_train_mlp, y_train, 
            X_valid_mlp, y_valid, 
            X_test_mlp, y_test,
            model_path=mlp_path,
            prev_model_1_path=ae1_path,
            prev_model_2_path=ae2_path,
            code_size_1=code_size_1,
            code_size_2=code_size_2
        )

        # 收集当前折的结果
        all_y_test.extend(y_test_fold)
        all_y_pred.extend(y_pred_fold)

    return np.array(all_y_test), np.array(all_y_pred)    


if __name__ == "__main__":
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, False)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)]) 
        except RuntimeError as e:
            print(e)
    
    start_time = time.time()
    
    reset()
    args = docopt(__doc__)
    pheno = load_phenotypes("./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv")
    hdf5 = hdf5_handler(b"./data/abide_single_site_fold5.hdf5", 'a')

    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160","cc400"]
    derivatives = [d for d in args["<derivative>"] if d in valid_derivatives]
    experiments = []
    for d in derivatives:
        cfg = {"derivative": d}
        if args["--child"]: experiments += [format_config("{derivative}_age_child", cfg)]
        if args["--teen"]: experiments += [format_config("{derivative}_age_teen", cfg)]
        if args["--adult"]: experiments += [format_config("{derivative}_age_adult", cfg)]
        if args["--whole"]: experiments += [format_config("{derivative}_whole", cfg)]
        if args["--male"]: experiments += [format_config("{derivative}_male", cfg)]
        if args["--threshold"]: experiments += [format_config("{derivative}_threshold", cfg)]
        if args["--leave-site-out"]:
            for site in pheno["SITE_ID"].unique():
                if site == 'NYU':
                    site_cfg = {"site": site}
                    experiments += [format_config("{derivative}_leavesiteout-{site}", cfg, site_cfg)]
        # 添加对 --single-site 的处理
        if args["--single-site"]:
            for site in pheno["SITE_ID"].unique():
                site_cfg = {"site": site}
                experiments += [format_config("{derivative}_singlesite_{site}", cfg, site_cfg)]
         # 添加对 --cmu-site 的处理（只跑 CMU）
        if args["--cmu-site"]:
            for site in pheno["SITE_ID"].unique():
                if site == "CMU":
                    site_cfg = {"site": site}
                    experiments += [format_config("{derivative}_singlesite_{site}", cfg, site_cfg)]

    code_size_1 = 300
    code_size_2 = 200
    y_pred = []
    turn = 0

    for experiment in sorted(experiments):
        print(f"\n=== Running experiment: {experiment} ===")
        y_test, y_pred = run_nn(hdf5, experiment, code_size_1, code_size_2)


    acc = np.mean(np.equal(np.argmax(y_pred, 1), np.argmax(y_test, 1)))
    print(f"\nFinal Accuracy: {acc:.4f}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")