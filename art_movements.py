import tensorflow as tf
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt

# class for artificial intelligence model
class Ai:
    def __init__(self):

        # const
        self.train_path = "../dataset/data/train"
        self.val_path = "../dataset/data/val"
        self.train_batch = 64
        self.val_batch = 32
        self.optimizer = "adam"


        # images
        self.train, self.val = self.image_creator()

        # gpu support
        self.gpu = tf.config.list_physical_devices("GPU")[0]
        tf.config.experimental.set_memory_growth(self.gpu, True)

        # image creator.

    def image_creator(self):

        train = ImageDataGenerator(rescale=1./255,
                                   rotation_range=45,
                                   fill_mode="nearest",
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   shear_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=True)
        val = ImageDataGenerator(rescale=1./255,
                                 rotation_range=45,
                                 fill_mode="nearest",
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 zoom_range=0.2,
                                 shear_range=0.1,
                                 horizontal_flip=True,
                                 vertical_flip=True)

        train_main = train.flow_from_directory(directory=self.train_path,
                                               batch_size=self.train_batch,
                                               shuffle=True,
                                               class_mode="categorical",
                                               target_size=(128, 128))

        val_main = val.flow_from_directory(directory=self.val_path,
                                           batch_size=self.val_batch,
                                           shuffle=True,
                                           class_mode="categorical",
                                           target_size=(128, 128))

        return train_main, val_main

    def model_cr(self):

        # model
        with tf.device("/GPU:0"):

            # model

            model = Sequential()

            # conv1
            model.add(Conv2D(64, (3, 3), padding="same", input_shape=(128, 128, 3), strides=(1, 1)))
            model.add(ReLU())
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

            # conv2
            model.add(Conv2D(128, (3, 3), padding="same"))
            model.add(ReLU())
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

            # conv3
            model.add(Conv2D(256, (3, 3), padding="same"))
            model.add(ReLU())
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(Conv2D(512, (3, 3), padding="same"))
            model.add(ReLU())
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

            # conv3
            model.add(Conv2D(1024, (3, 3), padding="same"))
            model.add(ReLU())
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(Conv2D(2048, (3, 3), padding="same"))
            model.add(ReLU())
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))


            # conv 4
            model.add(Conv2D(1024, (3, 3), padding="same"))
            model.add(ReLU())
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))



            # flatten
            model.add(Flatten())
            model.add(Dropout(0.1))

            # dense
            model.add(Dense(4096, activation="relu"))
            model.add(Dropout(0.1))

            # dense
            model.add(Dense(2048, activation="relu"))
            model.add(Dropout(0.1))

            # dense
            model.add(Dense(1024, activation="relu"))
            model.add(Dropout(0.1))

            model.add(Dense(13, activation="softmax"))

            # compile
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

            # summary
            model.summary()

            # train
            history = model.fit(self.train, validation_data=self.val,
                                epochs=150, )

            model.save("yeni_deneme.h5")

            # graphs
            h_pd = pd.DataFrame(history.history).plot()
            plt.show()



c = Ai()
c.model_cr()

