import tensorflow as tf
import SinGAN
import cv2
from config import config

tf.logging.set_verbosity(tf.logging.ERROR)

image_path = "Input/Images/"
input_name = "balloons.png"
image = cv2.imread(image_path + input_name)

singan = SinGAN.SinGAN(config = config, training_image = image, model = input_name.split(".")[0])

singan.random_samples(scale = 1)
singan.random_samples(scale = 2)
singan.random_samples(scale = 3)
singan.random_samples(scale = 4)
singan.random_samples(scale = 5)
singan.random_samples(scale = 6)
singan.random_samples(scale = 7)
singan.random_samples(scale = 8)
