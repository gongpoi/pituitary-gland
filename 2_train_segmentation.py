
import sys
import json
import os
import tensorflow as tf

from src.Config import Config
from src.Generator import Generator
from src.Model import Model

config_file = sys.argv[1]
dataset_file = sys.argv[2]
model_folder = sys.argv[3]
if not os.path.exists(model_folder): os.mkdir(model_folder)

config = Config(config_file)
generator = Generator(dataset_file,True,config)
val_generator = Generator(dataset_file,False,config)
model = Model(config).segmentation_model()
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_folder, 'segmentation.weights.h5'), save_best_only=True, save_weights_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_dice_coef_gland',
    mode='max',
    patience=10,
    min_delta=1e-4,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_dice_coef_gland', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

print(model.summary())
print(model.summary())

history = model.fit(generator,validation_data=val_generator,epochs=config.EPOCHS_SEGMENTATION,callbacks=[model_checkpoint_callback])

with open(os.path.join(model_folder, 'train-history-segmentation.json'), 'w') as outfile:
    json.dump(history.history, outfile)