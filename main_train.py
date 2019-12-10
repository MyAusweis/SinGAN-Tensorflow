import tensorflow as tf
import SinGAN
import cv2
from config import config

tf.logging.set_verbosity(tf.logging.ERROR)

image_path = "Input/Images/"
input_name = "balloons0.png"
image = cv2.imread(image_path + input_name)

singan = SinGAN.SinGAN(config = config, training_image = image, model = input_name.split(".")[0])
singan.train()
