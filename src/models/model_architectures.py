# src/models/model_architectures.py

import tensorflow as tf
import os
from tensorflow.keras.applications import (
    VGG16, EfficientNetB2, Xception, InceptionResNetV2, DenseNet201, NASNetMobile
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, GlobalAveragePooling2D, Dropout, Concatenate, Average
)

def get_base_model_class(model_name):
    """
    Returns the Keras application class from a string name.
    """

    model_map = {
        "VGG16": VGG16,
        "EfficientNetB2": EfficientNetB2,
        "Xception": Xception,
        "InceptionResNetV2": InceptionResNetV2,
        "DenseNet201": DenseNet201,
        "NASNetMobile": NASNetMobile
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown base model name: {model_name}")
    return model_map[model_name]

def build_base_model(model_config, training_config, data_config):
    """
    Builds a single model with a classification head.
    """
    
    base_model_fn = get_base_model_class(model_config.app_fn)
    img_shape = tuple(data_config.img_size) + (3,)
    
    base_model = base_model_fn(
        include_top=False,
        weights='imagenet',
        input_shape=img_shape
    )
    base_model.trainable = False

    inputs = Input(shape=img_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(training_config.optimal_hyperparameters.dropout_rate)(x)
    x = Dense(units=training_config.optimal_hyperparameters.dense_units, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name=model_config.name)
    return model

def find_gap_layer(model):
    """
    Recursively finds the GlobalAveragePooling2D layer.
    """
    
    for layer in reversed(model.layers):
        if isinstance(layer, GlobalAveragePooling2D):
            return layer
        if hasattr(layer, 'layers'):
            found = find_gap_layer(layer)
            if found: return found
    return None

def build_averaging_ensemble(base_models, preproc_fns, variant_name, data_config):
    """
    Builds an averaging ensemble from a list of trained models.
    """
    
    img_shape = tuple(data_config.img_size) + (3,)
    ensemble_input = Input(shape=img_shape, name=f"{variant_name}_ensemble_input")
    
    for model in base_models: # freeze base models
        model.trainable = False
        
    outputs = [model(fn(ensemble_input)) for model, fn in zip(base_models, preproc_fns)]
    avg_output = Average()(outputs)
    
    ensemble_model = Model(inputs=ensemble_input, outputs=avg_output, name=f'Averaging_Ensemble_{variant_name}')
    return ensemble_model

def build_stacking_ensemble(base_models, preproc_fns, variant_name, training_config, data_config):
    """
    Builds the stacking ensemble with a meta-learner.
    """
    
    img_shape = tuple(data_config.img_size) + (3,)
    ensemble_input = Input(shape=img_shape, name=f"{variant_name}_ensemble_input")

    feature_extractors = []
    for model in base_models:
        gap_layer = find_gap_layer(model)
        if not gap_layer:
            raise ValueError(f"Could not find GlobalAveragePooling2D layer in model {model.name}")
        extractor = Model(inputs=model.inputs, outputs=gap_layer.output)
        extractor.trainable = False
        feature_extractors.append(extractor)

    feature_outputs = [
        extractor(fn(ensemble_input)) for extractor, fn in zip(feature_extractors, preproc_fns)
    ]
    
    concatenated_features = Concatenate()(feature_outputs) if len(feature_outputs) > 1 else feature_outputs[0]
    
    # Meta-learner head
    x = Dropout(0.5)(concatenated_features)
    x = Dense(training_config.optimal_hyperparameters.dense_units, activation='relu')(x)
    x = Dropout(0.3)(x)
    stack_output = Dense(1, activation='sigmoid')(x)
    
    stacking_model = Model(inputs=ensemble_input, outputs=stack_output, name=f'Stacking_Ensemble_{variant_name}')
    return stacking_model