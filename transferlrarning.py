#######VGG16
import datetime
import tensorflow as tf
import keras
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard ,  LambdaCallback
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

!pip install numba

from numba import cuda
device = cuda.get_current_device()
device.reset()

from google.colab import drive
drive.mount('/content/drive')

dataset_dir = '/content/drive/MyDrive/Dataset'
Batch_size = 32
img_h = 224
img_w = 224
image_size = (img_h , img_w)
num_classes = 2
SEED = 64
classes = [
    'AD',
    'NC'
]

train_dir = os.path.join(dataset_dir , 'train')
val_dir = os.path.join(dataset_dir , 'valid')
test_dir = os.path.join(dataset_dir , 'test')
train_dataset = tf.keras.utils.image_dataset_from_directory( train_dir, labels ='inferred' , label_mode= 'categorical'
                                                            , shuffle=True , color_mode='rgb' , batch_size= Batch_size
                                                             ,seed = SEED , image_size = (img_h , img_w) )

valid_dataset = tf.keras.utils.image_dataset_from_directory( val_dir, labels ='inferred' , label_mode= 'categorical'
                                                           , shuffle=True , color_mode='rgb' , batch_size= Batch_size
                                                             ,seed = SEED , image_size = (img_h , img_w) )

test_dataset = tf.keras.utils.image_dataset_from_directory( test_dir, labels ='inferred' , label_mode= 'categorical'
                                                            , shuffle=True , color_mode='rgb' , batch_size= Batch_size
                                                             ,seed = SEED , image_size = (img_h , img_w) )
print( train_dataset.class_names)

data_normalize =  keras.Sequential(
    [
     keras.layers.Rescaling(1.0 / 255),
     keras.layers.Resizing(img_h , img_w)
    ])

train_dataset = train_dataset.map(
    lambda img, label: (data_normalize(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.map(
    lambda img, label: (data_normalize(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(
    lambda img, label: (data_normalize(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

#from keras.src.backend import flatten
from keras import layers
pre_train_model = tf.keras.applications.VGG16( weights= 'imagenet',
                                              include_top= False , input_shape= (img_h , img_w , 3))

for lyr in pre_train_model.layers[:]:
    lyr.trainable = False

base_input = pre_train_model.layers[1].input
base_output = pre_train_model.layers[-10].output
pool1 = keras.layers.MaxPooling2D()(base_output)
drop1 = layers.Dropout(0.5)(pool1)
conv_out_1 = layers.SeparableConv2D(64, 3, padding="same")(drop1)
batch_out_1 = layers.BatchNormalization()(conv_out_1)
act_out_1 = layers.Activation("relu")(batch_out_1)
pool_out = layers.GlobalAveragePooling2D()(act_out_1)
drop2 = layers.Dropout(0.5)(pool_out)
FCfinal = keras.layers.Dense( 2 , activation = 'softmax')(drop2)


model = keras.models.Model(inputs=[base_input] , outputs=[FCfinal])
model.summary()
#keras.utils.plot_model( model , to_file='modelplot.png', show_shapes=True , show_layer_names=True)

"""flatten = keras.layers.Flatten()(pool1)
FC1 = keras.layers.Dense( 128 , activation= 'relu')(flatten)
DR1 =  keras.layers.Dropout(0.6)(FC1)
FC2 = keras.layers.Dense( 64 , activation= 'relu')(DR1)
DR2 =  keras.layers.Dropout(0.3)(FC2)
FCfinal = keras.layers.Dense( 2 , activation = 'softmax')(DR2)
model = keras.models.Model(inputs=[base_input] , outputs=[FCfinal])
"""

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')
#from keras.api._v2.keras import callbacks
lrr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0.000001,

)
epochs_num = 20
history = model.fit(
                  train_dataset ,
                  epochs=epochs_num,
                  validation_data=valid_dataset,
                  callbacks=[tensorboard,lrr]
)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs_num)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot( )
plt.plot (epochs_range , acc , label ='training accuracy')
plt.plot (epochs_range , val_acc , label = 'validation accuracy')
plt.legend(loc = 'upper left')
plt.title('training and validation accuracy' )


plt.subplot(1,2,2)
plt.plot( epochs_range , loss , label ='training loss')
plt.plot (epochs_range , val_loss , label = 'validation loss')
plt.legend(loc = 'upper left')
plt.title('training and validation loss' )
plt.show()

model.evaluate(test_dataset, steps= epochs_num , verbose="auto")

y_pred = []  # store predicted labels
y_true = []  # store true labels

for image_batch, label_batch in valid_dataset:
  y_true.append(np.argmax(label_batch , axis= -1))
  preds = model.predict(image_batch)
  y_pred.append(np.argmax(preds, axis = - 1))

correct_labels = tf.concat([item for item in y_true], axis = 0)
predicted_labels = tf.concat([item for item in y_pred], axis = 0)
confusion_matrix(predicted_labels, correct_labels)

print(
    f"Classification report for classifier {history}:\n"
    f"{sklearn.metrics.classification_report(correct_labels, predicted_labels)}\n"
)


disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(predicted_labels, correct_labels , display_labels=classes)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

#ROC
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

y_pred = []  # store predicted labels
y_true = []  # store true labels

for image_batch, label_batch in test_dataset:
  y_true.append(np.argmax(label_batch , axis= -1))
  preds = model.predict(image_batch)
  y_pred.append(np.argmax(preds, axis = - 1))

correct_labels = tf.concat([item for item in y_true], axis = 0)
predicted_labels = tf.concat([item for item in y_pred], axis = 0)

fpr , tpr , thresholds = roc_curve(correct_labels , predicted_labels)
auc_1 = auc(fpr , tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr , tpr , label='model (area = {:.3f})'.format(auc_1))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


####ResNet152


#imports
import datetime
import tensorflow as tf
import keras
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard ,  LambdaCallback
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

!pip install numba

from numba import cuda
device = cuda.get_current_device()
device.reset()

from google.colab import drive
drive.mount('/content/drive')

dataset_dir = '/content/drive/MyDrive/Dataset'
Batch_size = 32
img_h = 224
img_w = 224
image_size = (img_h , img_w)
num_classes = 2
SEED = 64
classes = [
    'AD',
    'NC'
]

train_dir = os.path.join(dataset_dir , 'train')
val_dir = os.path.join(dataset_dir , 'valid')
test_dir = os.path.join(dataset_dir , 'test')
train_dataset = tf.keras.utils.image_dataset_from_directory( train_dir, labels ='inferred' , label_mode= 'categorical'
                                                            , shuffle=True , color_mode='rgb' , batch_size= Batch_size
                                                             ,seed = SEED , image_size = (img_h , img_w) )

valid_dataset = tf.keras.utils.image_dataset_from_directory( val_dir, labels ='inferred' , label_mode= 'categorical'
                                                           , shuffle=True , color_mode='rgb' , batch_size= Batch_size
                                                             ,seed = SEED , image_size = (img_h , img_w) )

test_dataset = tf.keras.utils.image_dataset_from_directory( test_dir, labels ='inferred' , label_mode= 'categorical'
                                                            , shuffle=True , color_mode='rgb' , batch_size= Batch_size
                                                             ,seed = SEED , image_size = (img_h , img_w) )
print( train_dataset.class_names)

data_normalize =  keras.Sequential(
    [
     keras.layers.experimental.preprocessing.Rescaling(1.0 / 255),
     keras.layers.experimental.preprocessing.Resizing(img_h , img_w)
    ])

train_dataset = train_dataset.map(
    lambda img, label: (data_normalize(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.map(
    lambda img, label: (data_normalize(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(
    lambda img, label: (data_normalize(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

from keras.src.backend import flatten
from keras import layers
pre_train_model = tf.keras.applications.ResNet152( weights= 'imagenet',
                                              include_top= False , input_shape= (img_h , img_w , 3))

for lyr in pre_train_model.layers[:]:
    lyr.trainable = False

base_input = pre_train_model.layers[0].input
base_output = pre_train_model.layers[-30].output
pool1 = keras.layers.MaxPooling2D()(base_output)
drop1 = keras.layers.Dropout(0.5)(pool1)
conv_out_1 = layers.SeparableConv2D(128, 3, padding="same")(drop1)
batch_out_1 = layers.BatchNormalization()(conv_out_1)
act_out_1 = layers.Activation("relu")(batch_out_1)
pool_out = layers.GlobalAveragePooling2D()(act_out_1)
drop2 = layers.Dropout(0.5)(pool_out)
FCfinal = keras.layers.Dense( 2 , activation = 'softmax')(drop2)


model = keras.models.Model(inputs=[base_input] , outputs=[FCfinal])
model.summary()
keras.utils.plot_model( model , to_file='modelplot.png', show_shapes=True , show_layer_names=True)

"""flatten = keras.layers.Flatten()(pool1)
FC1 = keras.layers.Dense( 128 , activation= 'relu')(flatten)
DR1 =  keras.layers.Dropout(0.6)(FC1)
FC2 = keras.layers.Dense( 64 , activation= 'relu')(DR1)
DR2 =  keras.layers.Dropout(0.3)(FC2)
FCfinal = keras.layers.Dense( 2 , activation = 'softmax')(DR2)
model = keras.models.Model(inputs=[base_input] , outputs=[FCfinal])
"""

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')
from keras.api._v2.keras import callbacks
lrr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0.000001,

)
epochs_num = 20
history = model.fit(
                  train_dataset ,
                  epochs=epochs_num,
                  validation_data=valid_dataset,
                  callbacks=[tensorboard,lrr]
)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs_num)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot( )
plt.plot (epochs_range , acc , label ='training accuracy')
plt.plot (epochs_range , val_acc , label = 'validation accuracy')
plt.legend(loc = 'upper left')
plt.title('training and validation accuracy' )


plt.subplot(1,2,2)
plt.plot( epochs_range , loss , label ='training loss')
plt.plot (epochs_range , val_loss , label = 'validation loss')
plt.legend(loc = 'upper left')
plt.title('training and validation loss' )
plt.show()

model.evaluate(test_dataset, steps= epochs_num , verbose="auto")

y_pred = []  # store predicted labels
y_true = []  # store true labels

for image_batch, label_batch in valid_dataset:
  y_true.append(np.argmax(label_batch , axis= -1))
  preds = model.predict(image_batch)
  y_pred.append(np.argmax(preds, axis = - 1))

correct_labels = tf.concat([item for item in y_true], axis = 0)
predicted_labels = tf.concat([item for item in y_pred], axis = 0)
confusion_matrix(predicted_labels, correct_labels)

print(
    f"Classification report for classifier {history}:\n"
    f"{sklearn.metrics.classification_report(correct_labels, predicted_labels)}\n"
)


disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(predicted_labels, correct_labels , display_labels=classes)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()



####InceptionResNetv2

#imports
import datetime
import tensorflow as tf
import keras
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard ,  LambdaCallback
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from google.colab import drive
drive.mount('/content/drive')

dataset_dir = '/content/drive/MyDrive/DataSet'
Batch_size = 32
img_h = 224
img_w = 224
image_size = (img_h , img_w)
num_classes = 2
SEED = 64
classes = [
    'AD',
    'NC'
]

train_dir = os.path.join(dataset_dir , 'train')
val_dir = os.path.join(dataset_dir , 'valid')
test_dir = os.path.join(dataset_dir , 'test')
train_dataset = tf.keras.utils.image_dataset_from_directory( train_dir, labels ='inferred' , label_mode= 'categorical'
                                                            , shuffle=True , color_mode="rgb"  , batch_size= Batch_size
                                                             ,seed = SEED , image_size = (img_h , img_w) )

valid_dataset = tf.keras.utils.image_dataset_from_directory( val_dir, labels ='inferred' , label_mode= 'categorical'
                                                           , shuffle=True , color_mode="rgb", batch_size= Batch_size
                                                             ,seed = SEED , image_size = (img_h , img_w) )

test_dataset = tf.keras.utils.image_dataset_from_directory( test_dir, labels ='inferred' , label_mode= 'categorical'
                                                            , shuffle=True , color_mode="rgb" , batch_size= Batch_size
                                                            ,seed = SEED , image_size = (img_h , img_w) )
print( train_dataset.class_names)

class_names =  train_dataset.class_names

plt.figure(figsize=(8, 8))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(np.argmax(labels[i]))
        plt.axis("off")

resize = keras.layers.Resizing(img_h , img_w)
rescale = keras.layers.Rescaling(1./255.0 , offset=0)


data_augmentation_layers = [
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomContrast(0.1),
    keras.layers.RandomZoom(0.05),
    keras.layers.Resizing(img_h , img_w),
    keras.layers.Rescaling(1./255.0 , offset=0)
]
def data_Normalization(images):
  return rescale(resize(images))

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

train_dataset = train_dataset.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.map(
    lambda img, label: (data_Normalization(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(
    lambda img, label: (data_Normalization(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

plt.figure(figsize=(8, 8))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(np.argmax(labels[i]))
        plt.axis("off")

from keras.src.backend import flatten
from keras import layers
pre_train_model = tf.keras.applications.InceptionResNetV2( weights= 'imagenet',
                                              include_top= False , input_shape= (img_h , img_w , 3))

for lyr in pre_train_model.layers[:]:
    lyr.trainable = False

base_input = pre_train_model.layers[0].input
base_output = pre_train_model.layers[-28].output
pool1 = keras.layers.MaxPooling2D()(base_output)
drop1 = layers.Dropout(0.5)(pool1)
conv_out_1 = layers.SeparableConv2D(128, 3, padding="same")(drop1)
batch_out_1 = layers.BatchNormalization()(conv_out_1)
act_out_1 = layers.Activation("relu")(batch_out_1)
pool_out = layers.GlobalAveragePooling2D()(act_out_1)
drop2 = layers.Dropout(0.5)(pool_out)
FCfinal = keras.layers.Dense( 2 , activation = 'softmax')(drop2)


model = keras.models.Model(inputs=[base_input] , outputs=[FCfinal])
model.summary()
keras.utils.plot_model( model , to_file='modelplot.png', show_shapes=True , show_layer_names=True)

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')
from keras.api._v2.keras import callbacks
lrr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0.000001,

)
epochs_num = 30
history = model.fit(
                  train_dataset ,
                  epochs=epochs_num,
                  validation_data=valid_dataset,
                  callbacks=[tensorboard,lrr]
)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs_num)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot( )
plt.plot (epochs_range , acc , label ='training accuracy')
plt.plot (epochs_range , val_acc , label = 'validation accuracy')
plt.legend(loc = 'upper left')
plt.title('training and validation accuracy' )


plt.subplot(1,2,2)
plt.plot( epochs_range , loss , label ='training loss')
plt.plot (epochs_range , val_loss , label = 'validation loss')
plt.legend(loc = 'upper left')
plt.title('training and validation loss' )
plt.show()

model.evaluate(test_dataset, steps= epochs_num , verbose="auto")

y_pred = []  # store predicted labels
y_true = []  # store true labels

for image_batch, label_batch in valid_dataset:
  y_true.append(np.argmax(label_batch , axis= -1))
  preds = model.predict(image_batch)
  y_pred.append(np.argmax(preds, axis = - 1))

correct_labels = tf.concat([item for item in y_true], axis = 0)
predicted_labels = tf.concat([item for item in y_pred], axis = 0)
confusion_matrix(predicted_labels, correct_labels)

print(
    f"Classification report for classifier {history}:\n"
    f"{sklearn.metrics.classification_report(correct_labels, predicted_labels)}\n"
)


disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(predicted_labels, correct_labels , display_labels=classes)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()