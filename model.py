#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np

#Denoising autoencoder
def ae(input_size, code_size,
       corruption=0.0, tight=False,
       enc=tf.nn.tanh, dec=tf.nn.tanh):
    """

    Autoencoder model: input_size -> code_size -> input_size
    Supports tight weights and corruption.

    """
    tf.compat.v1.disable_eager_execution()
    x = tf.placeholder(tf.float32, [None, input_size],name = "ae_input")

    if corruption > 0.0:
        # Corrupt data based on random sampling
        _x = tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                                      minval=0,
                                                      maxval=1 - corruption,
                                                      dtype=tf.float32), tf.float32))

    else:
        _x = x
    # Initialize encoder bias
    b_enc = tf.Variable(tf.zeros([code_size]),name="b_enc")

    # Initialize encoder weights using Glorot method
    W_enc = tf.Variable(tf.random_uniform(
                [input_size, code_size],
                -6.0 / math.sqrt(input_size + code_size),#使用 Glorot 初始化方法。
                6.0 / math.sqrt(input_size + code_size)),name="W_enc"
            )

    encode = tf.matmul(_x, W_enc) + b_enc
    if enc is not None:
        encode = enc(encode)

    b_dec = tf.Variable(tf.zeros([input_size]),name="b_dec")
    if tight:
        # Tightening using encoder weights
        W_dec = tf.transpose(W_enc,name="W_dec_transposed")

    else:
        # Initialize decoder weights using Glorot method
        W_dec = tf.Variable(tf.random_uniform(
                    [code_size, input_size],
                    -6.0 / math.sqrt(code_size + input_size),
                    6.0 / math.sqrt(code_size + input_size)),name="W_dec"
                )

    decode = tf.matmul(encode, W_dec) + b_dec
    if dec is not None:
        decode = enc(decode)

    model = {

        "input": x,

        "encode": encode,

        "decode": decode,

        "cost": tf.sqrt(tf.reduce_mean(tf.square(x - decode))),

        "params": {
            "W_enc": W_enc,
            "b_enc": b_enc,
            "b_dec": b_dec,
        }

    }

    if not tight:
        model["params"]["W_dec"] = W_dec

    return model

#Simplified variational encoder
def vae(input_size, latent_dim, kl_weight_ph, enc=tf.nn.tanh, dec=tf.nn.tanh):
    x = tf.placeholder(tf.float32, [None, input_size], name="vae_input")
    
    training = tf.placeholder_with_default(True, shape=(), name="training")

    # =====Encoder (single hidden layer + shared parameters to generate μ and logvar)=====
    with tf.variable_scope("encoder"):
        h1 = tf.layers.dense(x, 400, activation=enc, name="encoder_h1")
        h1_bn = tf.layers.batch_normalization(h1,training=training,name="encoder_h1_bn")
        x_proj = tf.layers.dense(x, 400, activation=None, name="proj_x") 
        h1_res = x_proj + h1_bn

        mu_logvar_layer = tf.layers.dense(
            h1_res, 
            2 * latent_dim, 
            activation=None,
            name="mu_logvar_layer"
        )
        z_mean = mu_logvar_layer[:, :latent_dim]
        z_log_var = mu_logvar_layer[:, latent_dim:]

    epsilon = tf.random_normal(tf.shape(z_log_var), name="epsilon")
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    z = tf.identity(z, name="latent_vector")

    with tf.variable_scope("decoder"):
        d1 = tf.layers.dense(z, 400, activation=dec, name="decoder_d1")
        d1_bn = tf.layers.batch_normalization(d1,training=training, name="decoder_d1_bn") # <=== 添加 training=
        z_proj = tf.layers.dense(z, 400, activation=None, name="proj_z")
        d1_res = z_proj + d1_bn
        x_recon = tf.layers.dense(d1_res, input_size, activation=None, name="x_recon")

    recon_loss = tf.reduce_mean(tf.square(x - x_recon))
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
    total_loss = recon_loss + kl_weight_ph * tf.reduce_mean(kl_loss)

    return {
        "input": x,
        "z_mean": z_mean,
        "z_log_var": z_log_var,
        "decode": x_recon,
        "cost": total_loss,
        "training": training,
        "params": tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    }

# Multilayer Neural Network
# With phenotypic data
def nn(input_size, n_classes, layers, pheno_size, init=None):
    tf.compat.v1.disable_eager_execution()

    input_fc = tf.placeholder(tf.float32, [None, input_size], name="mlp_input_fc")
    input_pheno = tf.placeholder(tf.float32, [None, pheno_size], name="mlp_input_pheno")
    y = tf.placeholder("float", [None, n_classes], name="expected")

    actvs = []
    dropouts = []
    params = {}
    regularizers = []
    x = input_fc

    for i, layer in enumerate(layers[:-1]):
        dropout = tf.placeholder(tf.float32, name=f"dropout_{i+1}")

        if init is None:
            W = tf.Variable(
                tf.random_uniform(
                    [input_size, layer["size"]],
                    -6.0 / math.sqrt(input_size + layer["size"]),
                    6.0 / math.sqrt(input_size + layer["size"]),
                ),
                name=f"W_{i+1}"
            )
            regularizers.append(tf.nn.l2_loss(W) * 0.01)
            b = tf.Variable(tf.zeros([layer["size"]]), name=f"b_{i+1}")
        else:
            W = tf.Variable(init[i]["W"], name=f"W_{i+1}")
            b = tf.Variable(init[i]["b"], name=f"b_{i+1}")

        x = tf.matmul(x, W) + b
        x = tf.keras.layers.BatchNormalization(name=f"bn_{i+1}")(x, training=True)
        
        if "actv" in layer and layer["actv"] is not None:
            x = layer["actv"](x)
        
        x = tf.nn.dropout(x, rate=1 - dropout)
        
        params.update({f"W_{i+1}": W, f"b_{i+1}": b})
        actvs.append(x)
        dropouts.append(dropout)
        input_size = layer["size"]

    last_layer = layers[-1]
    W_last = tf.Variable(
        tf.random_uniform(
            [input_size, last_layer["size"]],
            -6.0 / math.sqrt(input_size + last_layer["size"]),
            6.0 / math.sqrt(input_size + last_layer["size"]),
        ),
        name="W_last"
    )
    b_last = tf.Variable(tf.zeros([last_layer["size"]]), name="b_last")
    x = tf.matmul(x, W_last) + b_last
    x = tf.keras.layers.BatchNormalization(name="bn_last")(x, training=True)
    if "actv" in last_layer and last_layer["actv"] is not None:
        x = last_layer["actv"](x)
    x = tf.nn.dropout(x, rate=1 - dropouts[-1])

    x_combined = tf.concat([x, input_pheno], axis=1)

    W_out = tf.Variable(
        tf.random_uniform(
            [last_layer["size"] + pheno_size, n_classes],
            -3.0 / math.sqrt(last_layer["size"] + pheno_size + n_classes),
            3.0 / math.sqrt(last_layer["size"] + pheno_size + n_classes)
        ),
        name="W_out"
    )
    b_out = tf.Variable(tf.zeros([n_classes]), name="b_out")
    y_hat = tf.add(tf.matmul(x_combined, W_out), b_out, name="logits")

    regularizers.append(tf.nn.l2_loss(W_out) * 0.01)
    base_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y))
    reg_loss = tf.add_n(regularizers)
    total_loss = base_loss + reg_loss

    params.update({"W_last": W_last, "b_last": b_last, "W_out": W_out, "b_out": b_out})
    actvs.append(y_hat)

    return {
        "input_fc": input_fc,
        "input_pheno": input_pheno,
        "expected": y,
        "output": tf.nn.softmax(y_hat),
        "latent": x_combined,
        "cost": total_loss,
        "dropouts": dropouts,
        "actvs": actvs,
        "params": params,
    }
