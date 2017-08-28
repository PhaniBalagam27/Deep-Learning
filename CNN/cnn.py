# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()
# Step1 - Convolution
classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation='relu'))
# step2 - max pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))
 # step 3: flattening
 classifier.add(Flatten())
 #step4: full connection
 classifier.add(Dense(output_dim = 128, activation = 'relu'))
 classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

######################## part2 ##############################
# Fitting the CNN to images

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)


