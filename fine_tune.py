import os
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

import utilities as util
import finetuning_utils as ftu

from tensorflow.keras import layers

# ----------------------------
# Configurations
# ----------------------------
csv_file = "dataClassified/data.csv"
prepend_path = "dataClassified/"
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'

# ----------------------------
# Functions
# ----------------------------

def load_datasets(csv_file, prepend_path):
    """Load train, validation, and test datasets."""
    dataset_train = util.CustomDataset(csv_file, prepend_path, "train")
    dataset_val = util.CustomDataset(csv_file, prepend_path, "val")
    dataset_test = util.CustomDataset(csv_file, prepend_path, "test")
    return dataset_train, dataset_val, dataset_test


def load_yamnet_model(model_handle):
    """Load YAMNet model from TensorFlow Hub."""
    return hub.load(model_handle)


def preprocess_data(dataset, yamnet_model):
    """Preprocess dataset to extract embeddings and labels."""
    embeds, labels = ftu.preprocess_dataset(dataset, yamnet_model)
    return embeds, labels


def build_classifier(input_shape, num_classes=util.num_classes):
    """Build and compile the classifier model."""
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, embeds, labels, embeds_val, labels_val, epochs = util.num_epochs):
    """Train the classifier model."""
    history = model.fit(embeds, labels,
                        epochs=epochs,
                        validation_data=(embeds_val, labels_val))
    return history


def evaluate_model(model, embeds_test, labels_test):
    """Evaluate the trained model on test data."""
    test_loss, test_acc = model.evaluate(embeds_test, labels_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    return test_loss, test_acc


# ----------------------------
# Main Workflow
# ----------------------------

def main():
    # Load datasets
    dataset_train, dataset_val, dataset_test = load_datasets(csv_file, prepend_path)

    # Load YAMNet model
    yamnet_model = load_yamnet_model(yamnet_model_handle)

    # Preprocess datasets
    embeds, labels = preprocess_data(dataset_train, yamnet_model)
    embeds_val, labels_val = preprocess_data(dataset_val, yamnet_model)
    embeds_test, labels_test = preprocess_data(dataset_test, yamnet_model)

    # Build classifier
    model = build_classifier(input_shape=(1024,), num_classes=util.num_classes)

    # Train model
    history = train_model(model, embeds, labels, embeds_val, labels_val, epochs=util.num_epochs)

    # Evaluate model
    evaluate_model(model, embeds_test, labels_test)


if __name__ == "__main__":
    main()
