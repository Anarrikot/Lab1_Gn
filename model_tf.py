# model_tf.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_tf_regressor(input_dim, hidden=[128,64], dropout=0.1):
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for h in hidden:
        x = layers.Dense(h)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        if dropout>0:
            x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    return model
