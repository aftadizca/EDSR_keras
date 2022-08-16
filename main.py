import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
import logging
import os
import sys
import cv2

# disable gpu $env:DML_VISIBLE_DEVICES = -1
# os.environ["DML_VISIBLE_DEVICES"]="-1"
# print(os.environ["DML_VISIBLE_DEVICES"])
logging.getLogger("tensorflow").setLevel(logging.ERROR)


modelLoadPath = 'my_weight\edsrv4_1.h5'
modelSavePath = 'my_weight\edsrv4_1.h5'
learningRate = 25e-6

# Allow memory growth for the GPU
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print(physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

AUTOTUNE = tf.data.AUTOTUNE

div2k_data = tfds.image.Div2k(config="bicubic_x4")
div2k_data.download_and_prepare()

# Taking train data from div2k_data object
train = div2k_data.as_dataset(split="train", as_supervised=True)
train_cache = train.cache()

# Validation data
val = div2k_data.as_dataset(split="validation", as_supervised=True)
val_cache = val.cache()

full_model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='my_models',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

weight_only_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=modelSavePath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1)


def flip_left_right(lowres_img, highres_img):
    """Flips Images to left and right."""

    # Outputs random values from a uniform distribution in between 0 to 1
    rn = tf.random.uniform(shape=(), maxval=1)
    # If rn is less than 0.5 it returns original lowres_img and highres_img
    # If rn is greater than 0.5 it returns flipped image
    return tf.cond(
        rn < 0.5,
        lambda: (lowres_img, highres_img),
        lambda: (
            tf.image.flip_left_right(lowres_img),
            tf.image.flip_left_right(highres_img),
        ),
    )


def random_rotate(lowres_img, highres_img):
    """Rotates Images by 90 degrees."""

    # Outputs random values from uniform distribution in between 0 to 4
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    # Here rn signifies number of times the image(s) are rotated by 90 degrees
    return tf.image.rot90(lowres_img, rn), tf.image.rot90(highres_img, rn)


def random_crop(lowres_img, highres_img, hr_crop_size=192, scale=4):
    """Crop images.

    low resolution images: 24x24
    hight resolution images: 96x96
    """
    lowres_crop_size = hr_crop_size // scale  # 96//4=24
    lowres_img_shape = tf.shape(lowres_img)[:2]  # (height,width)

    lowres_width = tf.random.uniform(
        shape=(), maxval=lowres_img_shape[1] - lowres_crop_size + 1, dtype=tf.int32
    )
    lowres_height = tf.random.uniform(
        shape=(), maxval=lowres_img_shape[0] - lowres_crop_size + 1, dtype=tf.int32
    )

    highres_width = lowres_width * scale
    highres_height = lowres_height * scale

    lowres_img_cropped = lowres_img[
        lowres_height: lowres_height + lowres_crop_size,
        lowres_width: lowres_width + lowres_crop_size,
    ]  # 24x24
    highres_img_cropped = highres_img[
        highres_height: highres_height + hr_crop_size,
        highres_width: highres_width + hr_crop_size,
    ]  # 96x96

    return lowres_img_cropped, highres_img_cropped


def dataset_object(dataset_cache, training=True):

    ds = dataset_cache
    ds = ds.map(
        lambda lowres, highres: random_crop(lowres, highres, scale=4),
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
        ds = ds.map(flip_left_right, num_parallel_calls=AUTOTUNE)
    # Batching Data
    ds = ds.batch(16)

    if training:
        # Repeating Data, so that cardinality if dataset becomes infinte
        ds = ds.repeat()
    # prefetching allows later images to be prepared while the current image is being processed
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = dataset_object(train_cache, training=True)
val_ds = dataset_object(val_cache, training=False)

lowres, highres = next(iter(train_ds))

# # Hight Resolution Images
# plt.figure(figsize=(10, 10))
# for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(highres[i].numpy().astype("uint8"))
#     plt.title(highres[i].shape)
#     plt.axis("off")

# # Low Resolution Images
# plt.figure(figsize=(10, 10))
# for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(lowres[i].numpy().astype("uint8"))
#     plt.title(lowres[i].shape)
#     plt.axis("off")

# Using adam optimizer with initial learning rate as 1e-4, changing learning rate after 5000 steps to 5e-5
# optim_edsr = keras.optimizers.Adam(
#     learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(
#         boundaries=[1000], values=[1e-4, 5e-5]
#     )
# )

optim_edsr = keras.optimizers.Adam(
    learning_rate=learningRate,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
    amsgrad=False,
    name="Adam"
)


def PSNR(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    # Max value of pixel is 255
    psnr_value = tf.image.psnr(
        high_resolution, super_resolution, max_val=255)[0]
    return psnr_value


def SSIM(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    # Max value of pixel is 255
    ssim_value = tf.image.ssim(
        high_resolution, super_resolution, max_val=255)[0]
    return ssim_value


class EDSRModel(tf.keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, x):
        # Adding dummy dimension using tf.expand_dims and converting to float32 using tf.cast
        x = tf.cast(tf.expand_dims(x, axis=0), tf.float32)
        # Passing low resolution image to model
        super_resolution_img = self(x, training=False)
        # Clips the tensor from min(0) to max(255)
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 255)
        # Rounds the values of a tensor to the nearest integer
        super_resolution_img = tf.round(super_resolution_img)
        # Removes dimensions of size 1 from the shape of a tensor and converting to uint8
        super_resolution_img = tf.squeeze(
            tf.cast(super_resolution_img, tf.uint8), axis=0
        )
        return super_resolution_img


def expand(img_arr):
    return tf.cast(tf.expand_dims(img_arr, axis=0), tf.float32)


def squeeze(img_arr):
    img_arr = tf.clip_by_value(img_arr, 0, 255)
    # Rounds the values of a tensor to the nearest integer
    img_arr = tf.round(img_arr)
    # Removes dimensions of size 1 from the shape of a tensor and converting to uint8
    img_arr = tf.squeeze(
        tf.cast(img_arr, tf.uint8), axis=0
    )
    return img_arr

# Residual Block
def ResBlock(inputs):
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.Add()([inputs, x])
    # x = layers.Lambda(lambda t: t * 0.1)(x)
    return x

# Upsampling Block
def Upsampling(inputs, factor=2, **kwargs):
    x = layers.Conv2D(256 * (factor ** 2), 3, padding="same", **kwargs)(inputs)
    x = tf.nn.depth_to_space(x, block_size=factor)
    x = layers.Conv2D(256 * (factor ** 2), 3, padding="same", **kwargs)(x)
    x = tf.nn.depth_to_space(x, block_size=factor)
    return x


def make_model(num_filters, num_of_residual_blocks):
    # Flexible Inputs to input_layer
    input_layer = layers.Input(shape=(None, None, 3))
    # Scaling Pixel Values
    x = layers.Rescaling(scale=1.0 / 255)(input_layer)
    x = x_new = layers.Conv2D(num_filters, 3, padding="same")(x)

    # 16 residual blocks
    for _ in range(num_of_residual_blocks):
        x_new = ResBlock(x_new)

    x_new = layers.Conv2D(num_filters, 3, padding="same")(x_new)
    x = layers.Add()([x, x_new])

    x = Upsampling(x)
    x = layers.Conv2D(3, 3, padding="same")(x)

    output_layer = layers.Rescaling(scale=255)(x)
    return EDSRModel(input_layer, output_layer)


def train(epochs, modelpath=None):
    model = make_model(num_filters=256, num_of_residual_blocks=32)
    if modelpath:
        print('Load model weight')
        model.load_weights(modelpath)
    # model = keras.models.load_model("my_models", custom_objects={"PSNR":PSNR})
    # Compiling model with loss as mean absolute error(L1 Loss) and metric as psnr
    model.compile(optimizer=optim_edsr,
                  loss=tf.keras.losses.MeanAbsoluteError(), metrics=[PSNR])
    model.summary()
    # Training for more epochs will improve results
    model.fit(train_ds, epochs=epochs, steps_per_epoch=100,
              validation_data=val_ds, callbacks=[weight_only_callback])


def test(modelpath, input_img_path, output_img_path):
    model_load = make_model(num_filters=256, num_of_residual_blocks=32)
    model_load.load_weights(modelpath)
    model_load.summary()

    lowres = Image.open(input_img_path)
    lowres = lowres.convert('RGB')
    w, h = lowres.size
    
    lowres = np.array(lowres)
 
    preds = model_load.predict(lowres)

    image = Image.fromarray(preds)

    denoising_img = cv2.fastNlMeansDenoisingColored(
        np.array(preds), None, 3, 3, 7, 21)

    # image.save(output_img_path, format='PNG')
    cv2.imwrite(output_img_path,  cv2.cvtColor(denoising_img,
                cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])

    
if __name__ == '__main__':
    # train(epochs=50, modelpath=modelLoadPath)
    test(modelLoadPath, "E:\Images\WALLPAPER\\711l897jm4wz.jpg", 'output.png')
