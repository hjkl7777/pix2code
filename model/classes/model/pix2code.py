from __future__ import absolute_import
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

from keras.layers import Input, Dense, Dropout, \
                         RepeatVector, LSTM, concatenate, \
                         Conv2D, MaxPooling2D, Flatten,GRU,Permute,Multiply,Lambda,Add
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras import *
from .Config import *
from .AModel import *


class pix2code(AModel):
    def __init__(self, input_shape, output_size, output_path):
        AModel.__init__(self, input_shape, output_size, output_path)
        self.name = "pix2code"

        image_model = Sequential()
        image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=input_shape))
        image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
        image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Flatten())
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))

        image_model.add(RepeatVector(CONTEXT_LENGTH))
        image_model.summary()
        image_model2 = Sequential()
        image_model2.add(Conv2D(32, (7, 7), padding='valid', activation='relu', input_shape=input_shape))
        image_model2.add(Conv2D(32, (7, 7), padding='valid', activation='relu'))
        image_model2.add(MaxPooling2D(pool_size=(2, 2)))
        image_model2.add(Dropout(0.25))

        image_model2.add(Conv2D(64, (7, 7), padding='valid', activation='relu'))
        image_model2.add(Conv2D(64, (7, 7), padding='valid', activation='relu'))
        image_model2.add(MaxPooling2D(pool_size=(2, 2)))
        image_model2.add(Dropout(0.25))

        image_model2.add(Conv2D(128, (7, 7), padding='valid', activation='relu'))
        image_model2.add(Conv2D(128, (7, 7), padding='valid', activation='relu'))
        image_model2.add(MaxPooling2D(pool_size=(2, 2)))
        image_model2.add(Dropout(0.25))

        image_model2.add(Flatten())
        image_model2.add(Dense(1024, activation='relu'))
        image_model2.add(Dropout(0.3))
        image_model2.add(Dense(1024, activation='relu'))
        image_model2.add(Dropout(0.3))
        image_model2.add(RepeatVector(CONTEXT_LENGTH))
        image_model2.summary()




        visual_input = Input(shape=input_shape)
        encoded_image = image_model(visual_input)
        encoded_image2 = image_model2(visual_input)

        exchange_1 = Lambda(lambda x: x * 0.4)
        exchange_2 = Lambda(lambda x: x * 0.6)
        image_exchange_2 = exchange_2(encoded_image2)
        image_exchange_1 = exchange_1(encoded_image)

        encoded_image_end = Add()([image_exchange_1, image_exchange_2])


        language_model = Sequential()
        language_model.add(GRU(256, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        language_model.add(GRU(256, return_sequences=True))

        textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
        encoded_text = language_model(textual_input)

        encoded_text_permute = Permute((2, 1))(encoded_text)
        encoded_text_permute = Dense(48, activation='softmax')(encoded_text_permute)

        encoded_text_probs = Permute((2, 1))(encoded_text_permute)
        output_attention_mul = Multiply()([encoded_text, encoded_text_probs])
        language_model.summary()
        decoder = concatenate([encoded_image_end, output_attention_mul])

        decoder = GRU(512, return_sequences=True)(decoder)
        decoder = GRU(512, return_sequences=False)(decoder)
        decoder = Dense(output_size, activation='softmax')(decoder)

        self.model = Model(inputs=[visual_input, textual_input], outputs=decoder)

        optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        self.model.summary()
    def fit(self, images, partial_captions, next_words):
        self.model.fit([images, partial_captions], next_words, shuffle=False, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        self.save()

    def fit_generator(self, generator, steps_per_epoch):
        self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, verbose=1)
        self.save()

    def predict(self, image, partial_caption):
        return self.model.predict([image, partial_caption], verbose=0)[0]

    def predict_batch(self, images, partial_captions):
        return self.model.predict([images, partial_captions], verbose=1)
