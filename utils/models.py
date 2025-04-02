import os

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, BackupAndRestore
from tensorflow.keras import layers, models

def callbacks(model_name:str, path_models:str,
    reduce_patience:int,
    stop_patience:int,
    monitor:str,
    verbose:bool,
    mode:str,
    save_best_only:bool,
    save_weights_only:bool,
    restore_best_weights:bool,
    cooldown_epochs:int,
    lr:float,
    factor:float
    ):
    """ Manages the learning process of our model

    Args:
        model_name (str): model name
        path_models (str): path to save
        reduce_patience (int): decreases the lr when metrics doesn't change
        stop_patience (int): the number of epochs before the learning process is terminated if the metric doesn't change
        monitor (str): metric to monitor
        verbose (bool): shows the output
        mode (str): study mode
        save_best_only (bool): saves models with improved quality
        save_weights_only (bool): _description_
        restore_best_weights (bool): _description_
        cooldown_epochs (int): wait period when 
        lr (float): the learning rate
        factor (float): learning rate decrease factor (0,1)
    """

    # End training if the metric doesn't imporve
    earlystop = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=stop_patience,
        restore_best_weights=restore_best_weights,
        verbose=verbose
    )

    # Decrease learning rate if the metric doesn't improve 
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor, 
        mode=mode,  
        min_lr=lr/1000,
        factor=factor, 
        patience=reduce_patience,
        cooldown=cooldown_epochs,
        verbose=verbose
    )
    
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(path_models, model_name + '.weights.h5'), #full: '_model.keras' subclass:.h5
        #save_format="tf", # saves a subclassed model
        save_best_only=save_best_only, 
        save_weights_only=save_weights_only,
        monitor=monitor, 
        mode=mode,
        verbose=verbose        
    )
    
    backup = BackupAndRestore(
        # The path where your backups will be saved. Make sure the
        # directory exists prior to invoking `fit`.
        os.path.join(path_models),
        # How often you wish to save a checkpoint. Providing "epoch"
        # saves every epoch, providing integer n will save every n steps.
        save_freq="epoch",
        # Deletes the last checkpoint when saving a new one.
        delete_checkpoint=True,
    )
    # # reduces learnign rate smoothly
    # scheduler = LearningRateScheduler(
    #     schedule=smooth_decay(epoch, lr), 
    #     verbose=config.callbacks.verbose
    # )

    return [checkpoint, earlystop, reduce_lr, backup] 

class ModelVoice_v1(tf.keras.Model):
    def __init__(self, output_units:int=31, **kwargs):
        super(ModelVoice_v1, self).__init__(**kwargs)
        
        self.layers_list = [
            layers.Conv2D(32, (7, 3), activation='relu', padding="same", name='Conv2D_1'),
            layers.MaxPooling2D(pool_size=(1, 3), name='MaxPool2D_1'),
            layers.Conv2D(64, (1, 7), activation='relu', padding="same", name='Conv2D_2'),
            layers.MaxPooling2D(pool_size=(1, 4), name='MaxPool2D_2'),
            layers.Dropout(0.25, name='Dropout_1'),
            layers.Conv2D(128, (1, 10), activation='relu', padding="valid", name='Conv2D_3'),
            layers.Conv2D(256, (7, 1), activation='relu', padding="valid", name='Conv2D_4'),
            layers.GlobalMaxPooling2D(),
            layers.Dropout(0.5, name='Dropout_2'),
            layers.Dense(128, name='Dense_1'),
            layers.Dense(output_units, name='Output')]

    def call(self, inputs, training=False):
        x = inputs
        #x = self.input_layer(inputs)
        for layer in self.layers_list:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x

class ModelVoice_v2(tf.keras.Model):
    def __init__(self, output_units:int=31, **kwargs):
        super(ModelVoice_v2, self).__init__(**kwargs)
        #self.input_layer = layers.Input(shape=input_shape) # inferred automatically
        self.layers_list = [
            layers.Conv2D(64, (7, 3), activation='relu', padding='same', name='Conv2D_1'),
            layers.MaxPooling2D(pool_size=(1, 3), name='MaxPool2D_1'),
            layers.Conv2D(128, (1, 7), activation='relu', padding="same", name='Conv2D_2'),
            layers.MaxPooling2D(pool_size=(1, 4), name='MaxPool2D_2'),
            layers.Conv2D(256, (1, 10), activation='relu', padding="valid", name='Conv2D_3'),
            layers.Conv2D(512, (7, 1), activation='relu', padding="valid", name='Conv2D_4'),
            layers.GlobalMaxPooling2D(),
            layers.Dropout(0.5, name='Dropout_1'),
            layers.Dense(256, name='Dense_1'),
            layers.Dense(output_units, name='Output')
        ]
    
    def call(self, inputs, training=False):
        x = inputs
        #x = self.input_layer(inputs)
        for layer in self.layers_list:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x
