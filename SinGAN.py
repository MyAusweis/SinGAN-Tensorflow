import tensorflow as tf
import numpy as np
import math
import cv2
import models
import os

class SinGAN:
    def __init__(self, config, training_image, model):
        self.config = config
        self.full_image = training_image/127.5 -1
        self.name = model
        self.save_path = "Trained_models/" + model

        if not os.path.exists(self.save_path):
            print("Training from scratch")
            os.makedirs(self.save_path)

        self.graph = tf.Graph()
        self.__get_real_sizes()
        self.__build_graphs()


    def random_samples(self, scale = 0):
        with tf.Session(graph = self.graph) as sess:

            """
            Check if saved model exists

            """

            tf.global_variables_initializer().run()
            try:
                self.saver.restore(sess, self.save_path + "/model/model.ckpt")

            except:
                print("Error: Saved model not found")
                return


            """
            Check if random samples exists

            """

            save_path = "Output/RandomSamples/" + self.name + "/scale_" + str(scale)
            if not os.path.exists(save_path):
                os.makedirs(save_path)


            """
            Initialize input

            """

            NUM_SAMPLES = 50

            for j in range(NUM_SAMPLES):
                if scale == 0:
                    input = np.random.normal(0, self.config["noise_amp"], size = (1, self.real_pyramid[0].shape[0], self.real_pyramid[0].shape[1], self.config["nc_im"] ))
                else:
                    input = np.expand_dims(self.real_pyramid[scale], axis = 0)
                prev = input
                for i in range(scale, self.num_scales):

                    feed_dict = {self.random_samples_placeholders[i] : prev}
                    gen = sess.run(self.gen_random_samples[i], feed_dict = feed_dict)
                    if i < self.num_scales:
                        prev = np.expand_dims(cv2.resize(gen[0,:,:,:],  (self.real_pyramid[i+1].shape[1], self.real_pyramid[i+1].shape[0])), axis = 0)
                self.__save_image(prev, save_path + "/sample_" + str(j) + ".png")


    def train(self):
        with tf.Session(graph = self.graph) as sess:
            """
            Starts the training process

            """

            print("Training...")
            tf.global_variables_initializer().run()
            for scale in range(self.num_scales+1):
                for i in range(self.config["iter"]):


                    for j in range(self.config["Dsteps"]):

                        """
                        Optimize Discriminator

                        """
                        sess.run(self.discriminator_train_ops[scale])



                    for j in range(self.config["Gsteps"]):

                        """
                        Optimize Generator

                        """

                        sess.run(self.generator_train_ops[scale])

                    if i%25 == 1:
                        print("[Scale:" + str(scale) +"/" + str(self.num_scales) + "] Step:" + str(i) + "/" + str(self.config["iter"]))
                        gen, real = sess.run([self.generated_images[scale], self.tf_reals[scale]])
                        l = sess.run(self.generator_loss[scale])
                        print(l)
                        self.__save_image(gen, self.save_path + "/scale_" + str(scale) + ".png")
                        self.__save_image(real, self.save_path + "/real_" + str(scale) + ".png")
                        self.saver.save(sess, self.save_path + "/model/model.ckpt")


    def __save_image(self, image, path):
        image = image*127.5 + 127.5
        image = image[0,:,:,:]
        cv2.imwrite(path, image)

    def __build_graphs(self):
        """
        Build the computational graph

        """

        with self.graph.as_default():
            print("Building Graphs...")
            """
            Target images

            """

            self.tf_reals = []
            for i in range(self.num_scales + 1):
                self.tf_reals.append(tf.expand_dims(tf.Variable(self.real_pyramid[i], dtype = tf.float32, trainable = False, name = "scale_" + str(i) + "image"), axis = 0))


            """
            Generation graph

            """
            print("\tGenerator graphs...")

            self.generated_images = []

            first_noise = tf.random.normal(shape = (1, self.real_pyramid[0].shape[0], self.real_pyramid[0].shape[1], self.config["nc_im"]))
            prev = self.config["noise_amp"]*first_noise

            for i in range(self.num_scales + 1):
                generation = models.generator(prev, prev, self.config, name = "scale_" + str(i))
                self.generated_images.append(generation)

                if i != self.num_scales:
                    prev = tf.image.resize(self.tf_reals[i], (self.real_pyramid[i+1].shape[0], self.real_pyramid[i+1].shape[1]))
                    prev = prev + self.config["noise_amp"]*tf.random.normal(shape = tf.shape(prev))


            """
            Discriminator graph

            """

            print("\tDiscriminator graphs...")
            fake_discriminator_out = []
            for i in range(self.num_scales + 1):
                fake_discriminator_out.append(models.discriminator(self.generated_images[i], self.config, name = "scale_" + str(i)))

            real_discriminator_out = []
            for i in range(self.num_scales + 1):
                real_discriminator_out.append(models.discriminator(self.tf_reals[i], self.config, name = "scale_" + str(i)))



            """
            Generator loss

            """

            print("\tGenerator losses...")
            self.generator_loss = []

            for i in range(self.num_scales + 1):
                """
                Lrec

                """
                generated = self.generated_images[i]
                real = tf.image.resize(self.tf_reals[i], (tf.shape(generated)[1], tf.shape(generated)[2]))
                l_rec = tf.losses.mean_squared_error(labels = real, predictions = generated)

                """
                Gan Loss

                """
                l_gan = -tf.reduce_mean(fake_discriminator_out[i])

                # self.generator_loss.append(l_rec**0.5)
                self.generator_loss.append(l_gan + self.config['alpha']*l_rec)


            """
            Discriminator loss

            """

            print("\tDiscriminator losses...")
            self.discriminator_loss = []

            for i in range(self.num_scales + 1):
                """
                Gan loss

                """
                l_real = -tf.reduce_mean(real_discriminator_out[i])
                l_fake = tf.reduce_mean(fake_discriminator_out[i])

                """
                Gradient Penalty Loss

                """
                l_grad_penalty = self.__gradient_penalty_loss(real_data = self.tf_reals[i], fake_data = self.generated_images[i], LAMBDA = self.config["lambda_grad"], i = i)

                self.discriminator_loss.append(l_real + l_fake + l_grad_penalty)

            """
            Generator training op

            """

            print("\tGenerator training ops...")
            generator_optimzer = tf.train.AdamOptimizer(learning_rate = self.config["lr_g"], beta1 = self.config["beta1"], beta2 = 0.999)
            self.generator_train_ops = []

            for i in range(self.num_scales + 1):
                var_list = self.__get_vars("scale_" + str(i) + "/generator")
                self.generator_train_ops.append(generator_optimzer.minimize(self.generator_loss[i]))

            """
            Discriminator training op

            """

            print("\tDiscriminator training ops...")
            discriminator_optimizer = tf.train.AdamOptimizer(learning_rate = self.config["lr_d"], beta1 = self.config["beta1"], beta2 = 0.999)
            self.discriminator_train_ops = []

            for i in range(self.num_scales + 1):
                var_list = self.__get_vars("scale_" + str(i) + "/discriminator")
                self.discriminator_train_ops.append(discriminator_optimizer.minimize(self.discriminator_loss[i], var_list = var_list))

            """
            Random Samples

            """

            self.random_samples_placeholders = []
            self.gen_random_samples = []

            for i in range(self.num_scales):
                self.random_samples_placeholders.append(tf.placeholder(shape = (1, self.real_pyramid[i].shape[0], self.real_pyramid[i].shape[1], self.config["nc_im"] ),  dtype = tf.float32, name = "gen_input_" + str(i)))

            for i in range(self.num_scales):
                x = self.random_samples_placeholders[i] + self.config["noise_amp"]*tf.random.normal(shape = tf.shape(self.random_samples_placeholders[i]))
                gen = models.generator(x, x, self.config, name = "scale_" + str(i))
                self.gen_random_samples.append(gen)

            """
            Utils

            """
            self.saver = tf.train.Saver()

            print("Done")


    def __get_vars(self, str):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = str)

    def __gradient_penalty_loss(self, real_data, fake_data, LAMBDA, i):
        real_data = tf.image.resize(real_data, size = (tf.shape(fake_data)[1], tf.shape(fake_data)[2]))
        alpha = tf.random.uniform(shape = (1, 1))*tf.ones(shape = tf.shape(fake_data))

        interpolates = alpha * real_data + (tf.ones_like(alpha) - alpha)*fake_data
        disc_interpolates = models.discriminator(interpolates, self.config, name = "scale_" + str(i))
        gradients = tf.gradients(disc_interpolates, interpolates)[0]
        return tf.reduce_mean(gradients**2)*LAMBDA

    def __init_from_scratch(self):
        pass
        # self.sess.run(self.global_init)

    def __init_trained_model(self):
        pass

    def __get_real_sizes(self):
        self.num_scales = int(math.log(self.config["min_size"]*1.0/self.config["max_size"], self.config["scale_factor"]))
        width, height, _ = self.full_image.shape
        r = width/height

        real_sizes = []
        real_pyramid = []
        for i in range(self.num_scales+1):
            a = int(self.config["min_size"]*(self.config["scale_factor"])**(-i))
            b = int(a*max(r, 1/r))
            if r > 0:
                current_size = (b, a)
            else:
                current_size = (a, a)

            real_sizes.append(current_size)
            current_image = cv2.resize(self.full_image, current_size)
            real_pyramid.append(current_image)

        self.real_sizes = real_sizes
        self.real_pyramid = real_pyramid
