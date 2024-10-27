import keras
import tensorflow as tf
from glob import glob
import random, os, datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
import os
import random
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.preprocessing import image
from glob import glob
import random, os, datetime

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, metrics
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models  import Sequential, load_model
from tensorflow.keras.metrics import Accuracy, AUC

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def extract_zip(zip_path, extract_to):
    """Extract the ZIP file to the specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def get_image_paths(root_dir, num_images=None):
    """Get the paths of images in the specified directory."""
    all_images = []
    for extension in ['*.jpg', '*.jpeg', '*.png']:
        all_images.extend(glob(os.path.join(root_dir, '**', extension), recursive=True))
    if num_images is None:
        return all_images
    else:
        return random.sample(all_images, min(num_images, len(all_images)))

def display_images_with_labels(img_list, root_dir):
    """Display images in a grid format with their real class labels."""
    plt.figure(figsize=(15, 6))
    for i, img_path in enumerate(img_list):
        img = image.load_img(img_path)
        img = image.img_to_array(img, dtype=np.uint8)
        label = img_path.split('/')[-2]
        plt.subplot(2, 5, i + 1)
        plt.imshow(img.squeeze())
        plt.axis('off')
        plt.title(f'{label}')
    plt.tight_layout()
    plt.show()

zip_file_path = '/content/covid.zip'  
extract_to_dir = '/content/yeni deneme' 
extract_zip(zip_file_path,extract_to_dir)
reals = "/content/yeni deneme"

root_dir = reals

num_images_to_display = 10 
image_paths = get_image_paths(root_dir, num_images=num_images_to_display)  
display_images_with_labels(image_paths, root_dir)



dir_path="/content/yeni deneme/covid/train"


train = ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         validation_split=0.20,
                         rescale=1./255,
                         shear_range = 0.2,
                         zoom_range = 0.2,
                         width_shift_range = 0.2,
                         height_shift_range = 0.3,)

val = ImageDataGenerator(rescale=1/255,
                        validation_split=0.20)


train_generator=train.flow_from_directory(dir_path,
                                          target_size=(224, 224),
                                          batch_size=32,
                                          class_mode='categorical',
                                          subset='training')

validation_generator=val.flow_from_directory(dir_path,
                                        target_size=(224, 224),
                                        batch_size=32,
                                        class_mode='categorical',
                                        subset='validation')



from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import AUC

model = Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

  # 4 katmanlı nöronlar kullanılmıştır 3 sınıf olduğundan "softmax" ve "categorical_crossentropy" kullanılmıştır
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(3, activation='softmax')  
])

metrics = [
    "accuracy",
    AUC(name='auc')
]

model.compile(optimizer='adam',
              loss='categorical_crossentropy',  
              metrics=metrics)

early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=10,
                               mode='max',
                               verbose=1,
                               restore_best_weights=True)

model_checkpoint = ModelCheckpoint(filepath='covid.keras',
                                   monitor='val_accuracy',
                                   save_best_only=True,
                                   save_weights_only=False,
                                   verbose=1)
start_time = datetime.datetime.now()

history = model.fit(train_generator,
                    epochs=100,
                    validation_data=validation_generator,
                    callbacks=[early_stopping, model_checkpoint])

end_time = datetime.datetime.now()
total_duration = end_time - start_time
print("Trainin Time:", total_duration)



val_loss, val_accuracy, val_auc = model.evaluate(validation_generator, verbose=0)
print(f"Loss: {val_loss}")
print(f"Accuracy: {val_accuracy}")
print(f"AUC: {val_auc}")


Loss: 0.2961
Accuracy: 0.9200
AUC: 0.9793

waste_labels = {0: 'Covid', 1: 'Normal', 2: 'Viral Pneumonia'} # 'Viral Pneumonia =Viral Zatürre

im_dir = "/content/yeni deneme/covid/test"
covid_model = load_model('/content/covid.keras')
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)

    img_array = image.img_to_array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img, img_array
def prediction_probs(img_array, model, waste_labels):

    predictions = model.predict(img_array, verbose = 0)

    predicted_class_idx = np.argmax(predictions[0])

    predicted_class = waste_labels.get(predicted_class_idx, 'Unknown')

    max_probability = np.max(predictions[0])

    return max_probability, predicted_class
def display_images(image_paths, model, waste_labels):
    """Display images along with their predicted and true labels."""
    num_images = len(image_paths)
    num_cols = 4
    num_rows = (num_images + num_cols - 1) // num_cols
    plt.figure(figsize=(num_cols * 5, num_rows * 5))

    for i, path in enumerate(image_paths):
        img, img_array = preprocess_image(path)

        probability, predicted_class = prediction_probs(img_array, model, waste_labels)

        ax = plt.subplot(num_rows, num_cols, i + 1)
        img = image.img_to_array(img)
        plt.imshow(img.astype('uint8'))

        true_label = path.split('/')[-2]

        plt.title(f"Max Probability: {probability:.2f}\nPredicted Class: {predicted_class}\nTrue Class: {true_label}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
get_image_paths(im_dir, 20)
random_images_path = get_image_paths(im_dir, 20)
display_images(random_images_path, covid_model, waste_labels)

