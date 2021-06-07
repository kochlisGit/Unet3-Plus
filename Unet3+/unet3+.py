import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.activations as activations
import tensorflow.keras.metrics as metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons.optimizers as optimizers
import tensorflow_addons.losses as losses
import numpy as np
import cv2


# Loads dataset.
# Targets are the segmantation images.
# Inputs are the original images.
def load_dataset(dataset_path):
    data = np.load(dataset_path, allow_pickle=True)
    return data['inputs'], data['targets']


# Defining the encoder's down-sampling blocks.
def encoder_block(inputs, n_filters, kernel_size, strides):
    encoder = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(activations.gelu)(encoder)
    encoder = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same', use_bias=False)(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(activations.gelu)(encoder)
    return encoder


# Defining the decoder's up-sampling blocks.
def upscale_blocks(inputs):
    n_upscales = len(inputs)
    upscale_layers = []

    for i, inp in enumerate(inputs):
        p = n_upscales - i
        u = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2**p, padding='same')(inp)

        for i in range(2):
            u = layers.Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(u)
            u = layers.BatchNormalization()(u)
            u = layers.Activation(activations.gelu)(u)
            u = layers.Dropout(rate=0.4)(u)

        upscale_layers.append(u)
    return upscale_layers


# Defining the decoder's whole blocks.
def decoder_block(layers_to_upscale, inputs):
    upscaled_layers = upscale_blocks(layers_to_upscale)

    decoder_blocks = []

    for i, inp in enumerate(inputs):
        d = layers.Conv2D(filters=64, kernel_size=3, strides=2**i, padding='same', use_bias=False)(inp)
        d = layers.BatchNormalization()(d)
        d = layers.Activation(activations.gelu)(d)
        d = layers.Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(d)
        d = layers.BatchNormalization()(d)
        d = layers.Activation(activations.gelu)(d)

        decoder_blocks.append(d)

    decoder = layers.concatenate(upscaled_layers + decoder_blocks)
    decoder = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', use_bias=False)(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation(activations.gelu)(decoder)
    decoder = layers.Dropout(rate=0.4)(decoder)

    return decoder


def get_model(input_dim):
    inputs = layers.Input(input_dim)

    e1 = encoder_block(inputs, n_filters=32, kernel_size=3, strides=1)
    e2 = encoder_block(e1, n_filters=64, kernel_size=3, strides=2)
    e3 = encoder_block(e2, n_filters=128, kernel_size=3, strides=2)
    e4 = encoder_block(e3, n_filters=256, kernel_size=3, strides=2)
    e5 = encoder_block(e4, n_filters=512, kernel_size=3, strides=2)

    d4 = decoder_block(layers_to_upscale=[e5], inputs=[e4, e3, e2, e1])
    d3 = decoder_block(layers_to_upscale=[e5, d4], inputs=[e3, e2, e1])
    d2 = decoder_block(layers_to_upscale=[e5, d4, d3], inputs=[e2, e1])
    d1 = decoder_block(layers_to_upscale=[e5, d4, d3, d2], inputs=[e1])

    output = layers.Conv2D(filters=3, kernel_size=1, padding='same', activation='tanh')(d1)

    model = models.Model(inputs, output)
    return model


model = get_model((256, 256, 3))
model.summary()


# Loading the dataset.
# x_train: Segmentation images.
# y_train: Original images.
DATASET_PATH = 'D:\\Datasets\\City-Segmentation\\Cityscapes\\dataset.npz'
x_train, y_train = load_dataset(DATASET_PATH)

# Normalizing data.
x_train = (x_train - 127.5) / 127.5
y_train = (y_train - 127.5) / 127.5

# Building & Compiling the model.
image_size = x_train[0].shape
unet3_plus = get_model(image_size)

unet3_plus.compile(
    optimizer=optimizers.Yogi(learning_rate=0.00025),
    loss=losses.sigmoid_focal_crossentropy,
    metrics=[metrics.MeanIoU(num_classes=30)]
)

# Training the model
epochs = 200
batch_size = 4

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=x_train.shape[0])
inputs = dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
batches_per_epoch = x_train.shape[0] // batch_size

for epoch in range(epochs):
    print('\nTraining on epoch', epoch + 1)

    loss = 0
    for i, (x_batch, y_batch) in enumerate(inputs):
        loss = unet3_plus.train_on_batch(x_batch, y_batch)

        print('\rCurrent batch: {}/{} , loss = {}'.format(
            i+1,
            batches_per_epoch,
            loss, end='')
        )
