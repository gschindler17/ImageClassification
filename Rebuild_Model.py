# %%
# Running all of the imported packages
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import PIL
# Notice that this import takes a while
# This is amplified if using a virtual environment
print("Beginning to import tensorflow...")
import tensorflow as tf
print("tensorflow has been imported.")

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib


# %%
# Used for importing the dataset off of the web
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)

# Should print "data_dir: C:\Users\Garrett\.keras\datasets\flower_photos.tar"
print("data_dir: {}".format(data_dir))

data_dir = pathlib.Path(data_dir).with_suffix('')
# Should print "data_dir: C:\Users\Garrett\.keras\datasets\flower_photos"
print("data_dir: {}".format(data_dir))

image_data = list(data_dir.glob('*/*.jpg'))
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Number of images found: {}".format(image_count))


# %%
# Sets parameters for the loader
batch_size = 32
img_height = 180
img_width = 180

# %%
# Beginning the splitting
# It's good practice to use a validation split when developing your model. 
# Use 80% of the images for training and 20% for validation.
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# %%
# Finding the class names from the training set
class_names = train_ds.class_names
print(class_names)

# %%
# Printing out a 9x9 set of images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

print("Completed.")


# %%
# Configuring the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Configured.")





# %%
# Standardizing the data

# Changing the RGB range from [0, 255] to [0, 1] by using tf.keras.layers.Rescaling
normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))



# %%
# Creating the model
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])





# %%
# Compiling the Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# %%
# Printing the model summary
model.summary()


# %%
# Training the model
# Trains the model for 10 epochs with the Keras Model.fit method
# This set takes a long time
# epochs=10
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
# )


# %%
# Visualizing the training results
print("\nCode is commented out.\n")
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()




# %%
'''
End of the inital tutorial
Below this point is optimization to minimize overfitting

Data Augmentation and Dropout
'''







# %%
# Data augmentation; "creating" more samples to train model on

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)
    


# %%
# Visualizing the data augmentation

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")


# %% 
# Adding in Dropout to a new model "drop_model"

drop_model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])


# %%
# Compiling the drop_model network and training it
drop_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
     

drop_model.summary()
     

epochs = 14
history = drop_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# %%
# Visualizing the training results of the drop_model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()




# %%
# Prediction on a red sunflower using the model

sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = drop_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)





# %%
# Save the TensorFlow Lite Model to a file

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(drop_model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)


# %%
# Interpret using the TesorFlow Lite Model file
TF_MODEL_FILE_PATH = 'model.tflite' # The default path to the saved TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
interpreter.get_signature_list()


classify_lite = interpreter.get_signature_runner('serving_default')


sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions_lite = classify_lite(sequential_3_input=img_array)['outputs']
score_lite = tf.nn.softmax(predictions_lite)
     

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)
# print(np.max(np.abs(predictions - predictions_lite)))  
# %%





# # %%
# # Confusion matrix?
# import sklearn
# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
# sns.set_style('darkgrid')
# classes=test_ds.class_names # ordered list of class names
# ytrue=[]
# for images, label in test_ds:   
#     for e in label:
#         ytrue.append(classes[e]) # list of class names associated with each image file in test dataset 
# ypred=[]
# errors=0
# count=0
# preds=model.predict(test_ds, verbose=1) # predict on the test data
# for i, p in enumerate(preds):
#     count +=1
#     index=np.argmax(p) # get index of prediction with highest probability
#     klass=classes[index] 
#     ypred.append(klass)  
#     if klass != ytrue[i]:
#         errors +=1
# acc= (count-errors)* 100/count
# msg=f'there were {count-errors} correct predictions in {count} tests for an accuracy of {acc:6.2f} % '
# print(msg) 
# ypred=np.array(ypred)
# ytrue=np.array(ytrue)
# if len(classes)<= 30: # if more than 30 classes plot is not useful to cramed
#         # create a confusion matrix 
#         cm = confusion_matrix(y_true, y_pred )        
#         length=len(classes)
#         if length<8:
#             fig_width=8
#             fig_height=8
#         else:
#             fig_width= int(length * .5)
#             fig_height= int(length * .5)
#         plt.figure(figsize=(fig_width, fig_height))
#         sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)       
#         plt.xticks(np.arange(length)+.5, classes, rotation= 90)
#         plt.yticks(np.arange(length)+.5, classes, rotation=0)
#         plt.xlabel("Predicted")
#         plt.ylabel("Actual")
#         plt.title("Confusion Matrix")
#         plt.show()
# clr = classification_report(ytrue, ypred, target_names=class_names)
# print("Classification Report:\n----------------------\n", clr) 

# %%
