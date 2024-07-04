import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

data_dir = 'dataset_kaggle'
filepaths = []
labels = []

folds = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    for file in os.listdir(foldpath):
        filepaths.append(os.path.join(foldpath, file))
        if 'lungaca' in file:
            labels.append('Lung Adenocarcinoma')
        elif 'lungn' in file:
            labels.append('Lung Benign Tissue')
        elif 'lungscc' in file:
            labels.append('Lung Squamous Cell Carcinoma')

df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})

labels = df['labels']
train_df, temp_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=26, stratify=labels)
valid_df, test_df = train_test_split(temp_df, train_size=0.5, shuffle=True, random_state=26, stratify=temp_df['labels'])

batch_size = 32
img_size = (224, 224)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels',
                                              target_size=img_size, class_mode='categorical',
                                              color_mode='rgb', shuffle=True, batch_size=batch_size)

valid_gen = test_datagen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels',
                                             target_size=img_size, class_mode='categorical',
                                             color_mode='rgb', shuffle=False, batch_size=batch_size)

test_gen = test_datagen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels',
                                            target_size=img_size, class_mode='categorical',
                                            color_mode='rgb', shuffle=False, batch_size=batch_size)

classes = list(train_gen.class_indices.keys())
class_count = len(classes)

base_model = Xception(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # this freezes the base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(class_count, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint(filepath='model.Xception.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
]

history = model.fit(train_gen, epochs=20, validation_data=valid_gen, callbacks=callbacks)

test_loss, test_acc = model.evaluate(test_gen)
print(f"Test accuracy: {test_acc}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()