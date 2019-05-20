from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from trainer.utils import analyze_result, compute_confusion_matrix
from trainer.model import create_model
import pickle
from os.path import join, pardir
from constants import TRAIN_DATA_DIR, VALIDATION_DATA_DIR, TRAINED_MODEL_PATH, BATCH_SIZE, IMAGE_SIZE, EPOCHS

checkpoint = ModelCheckpoint("../data_files/model/model_weights.h5", monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint]


def create_train_data_generator():
    train_image_gen = ImageDataGenerator()
    train_generator = train_image_gen.flow_from_directory(
        directory=join(pardir, TRAIN_DATA_DIR),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True
    )
    return train_generator


def create_validation_data_generator():
    validation_image_gen = ImageDataGenerator()
    validation_generator = validation_image_gen.flow_from_directory(
        directory=join(pardir, VALIDATION_DATA_DIR),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    return validation_generator


def train_model():
    train_generator = create_train_data_generator()
    validation_generator = create_validation_data_generator()

    model = create_model()
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.n,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.n,
        callbacks=callbacks_list
    )

    with open(join(pardir, TRAINED_MODEL_PATH), 'wb') as file:
        pickle.dump(model, file)

    analyze_result(history)
    compute_confusion_matrix(model, validation_generator)


train_model()
