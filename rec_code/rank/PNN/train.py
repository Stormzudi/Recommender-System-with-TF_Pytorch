#!/usr/bin/env python
"""
Created on 05 11, 2022

model: train.py

@Author: Stormzudi
"""
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from rank.contrib.utils import create_criteo_dataset
from rank.PNN.model import PNN

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    # you can modify your file path
    file = "../dataset/CriteoTrain.txt"
    read_part = True
    sample_num = 6000000
    test_size = 0.2

    embed_dim = 8
    mode = 'in'
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 256
    epochs = 30
    # ========================== Create dataset =======================
    feature_columns, train, test = create_criteo_dataset(file=file,
                                                         embed_dim=embed_dim,
                                                         read_part=read_part,
                                                         sample_num=sample_num,
                                                         test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = PNN(feature_columns, hidden_units, dnn_dropout)
    model.summary()

    # ============================model checkpoint======================
    check_path = 'save/pnn_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
                                                    verbose=1, period=5)

    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])

    # ===========================Fit==============================
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', min_delta=0.01,
                                                      patience=1, verbose=0, mode='max',
                                                      baseline=None, restore_best_weights=True)
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping],
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])