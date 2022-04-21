#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[36]:


import os
import json
import random
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization


# ## Data Preprocessing

# In[20]:


file_path = r'C:\Users\Krishna Sahoo\Documents\Python Venv\HerokuDemo\Images'
filenames = os.listdir(file_path)


# In[21]:


categories = []

for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


# In[22]:


df.category = df.category.map(
    {
        0: '0',
        1: '1'
    }
)

df.head()


# In[23]:


sample = random.choice(filenames)
image = load_img(os.path.join(file_path, sample))
plt.imshow(image)


# In[24]:


train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


# In[25]:


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)


train_generator = train_datagen.flow_from_dataframe(
    train_df,
    file_path, 
    x_col='filename',
    y_col='category',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32
)


# In[26]:


train_df


# In[27]:


validate_df


# In[28]:


validation_datagen = ImageDataGenerator(
    rescale=1./255,
)

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    file_path, 
    x_col='filename',
    y_col='category',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32
)


# In[29]:


# Helper Function

def check(t):
  if t[0] == 1:
    return "Cat"
  
  return "Dog"


# In[30]:


plt.figure(figsize=(12, 12))

for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in validation_generator:
        image = X_batch[0]
        plt.xlabel("Actual: " + check(Y_batch[0]))
        plt.imshow(image)
        break

plt.tight_layout()
plt.show()


# ## Model Architecture

# In[31]:


model = Sequential()

model.add(Conv2D(64, kernel_size=5, kernel_initializer='he_uniform', padding='same', input_shape=(224, 224, 3)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=3, strides=2, kernel_initializer='he_uniform'))
model.add(Activation('elu'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64, kernel_size=3, kernel_initializer='he_uniform', padding='same'))
model.add(Activation('elu'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=5, strides=2, kernel_initializer='he_uniform', padding='same'))
model.add(Activation('elu'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(756, kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(256, kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(2, kernel_initializer='he_uniform'))
model.add(Activation('sigmoid'))

model.summary()


# In[32]:


model.compile(
    loss = 'binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])


# ## Model Training

# In[33]:


history = model.fit(
    train_generator, 
    epochs=10,
    validation_data=validation_generator,
)


# In[34]:


score = model.evaluate(validation_generator, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# summarize history for accuracy  
   
plt.subplot(211)  
plt.plot(history.history['accuracy'])  
plt.plot(history.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
  
# summarize history for loss  
   
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  

plt.subplots_adjust(right=3, top=3)
plt.show()


# In[37]:


tf.keras.models.save_model(model,'my_model.hdf5')


# ## Performance on Test Data

# In[ ]:


model.save('final.h5')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


import pandas as pd
import numpy as np

val_image_batch, val_label_batch = next(iter(validation_generator))
true_label_ids = np.argmax(val_label_batch, axis=-1)

pred = model.predict(val_image_batch)
pred_df = pd.DataFrame(pred)
pred_df.columns = [1, 0]
pred_ids = np.argmax(pred, axis=-1)


plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)
for n in range(15):
  plt.subplot(5, 3, n+1)
  plt.imshow(val_image_batch[n])

  plt.xlabel("Cat" if pred_ids[n] == 0 else "Dog")

