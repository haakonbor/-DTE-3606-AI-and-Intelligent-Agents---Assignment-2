import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pydot
import pydotplus
import graphviz
from keras.datasets.mnist import load_data
from keras import layers, models, optimizers
from keras.utils import plot_model


def assignment2():
    # Hyperparameters
    latent_space_dimensions = 100
    samples = 25

    # Discriminator
    discriminator_model = create_discriminator()
    # dataset = load_real_samples()
    # train_discriminator(discriminator_model, dataset)

    # Generator
    generator_model = create_generator(latent_space_dimensions)

    # GAN
    gan = create_gan(generator_model, discriminator_model)

    plt.show()


def load_real_samples():
    (train_x, _), (_, _) = load_data()
    x = np.expand_dims(train_x, axis=-1)     # Expand from 2D to 3D
    x = x.astype('float32')
    x = x / 255.0  # Normalize data to [0, 1]
    return x


def generate_real_samples(dataset, samples):
    random_index = np.random.randint(0, dataset.shape[0], samples)
    x = dataset[random_index]
    y = np.ones((samples, 1))
    return x, y


def generate_fake_samples(generator_model, latent_dimensions, samples):
    points = generate_latent_points(latent_dimensions, samples)
    x = generator_model.predict(points)
    y = np.zeros((samples, 1))
    return x, y


def generate_latent_points(latent_dimensions, samples):
    points = np.random.randn(latent_dimensions * samples)
    points = points.reshape(samples, latent_dimensions)
    return points


def create_gan(generator_model, discriminator_model):
    discriminator_model.trainable = False
    gan_model = models.Sequential()
    gan_model.add(generator_model)
    gan_model.add(discriminator_model)
    optimization = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    gan_model.compile(loss='binary_crossentropy', optimizer=optimization)

    gan_model.summary()
    plot_model(gan_model, to_file='gan_model.png', show_shapes=True, show_layer_names=True)
    return gan_model


def create_discriminator(input_structure=(28, 28, 1), structure='conv'):
    if structure == 'conv':
        return discriminator_conv(input_structure)
    else:
        return None


def create_generator(latent_space_dim, structure='conv'):
    if structure == 'conv':
        return generator_conv(latent_space_dim)
    else:
        return None


def generator_conv(latent_space_dim):
    generator_model = models.Sequential()

    # 7x7 image (foundation created from latent space)
    generator_model.add(layers.Dense(7 * 7 * 128, input_dim=latent_space_dim))
    generator_model.add(layers.LeakyReLU(alpha=0.2))
    generator_model.add(layers.Reshape((7, 7, 128)))

    # 14x14 up-sampled image
    generator_model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    generator_model.add(layers.LeakyReLU(alpha=0.2))

    # 28x28 up-sampled image
    generator_model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    generator_model.add(layers.LeakyReLU(alpha=0.2))

    generator_model.add(layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same'))

    generator_model.summary()
    plot_model(generator_model, to_file='generator_model.png', show_shapes=True, show_layer_names=True)

    return generator_model


def discriminator_conv(input_structure):
    discriminator_model = models.Sequential()
    discriminator_model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same',
                                          input_shape=input_structure))
    discriminator_model.add(layers.LeakyReLU(alpha=0.2))
    discriminator_model.add(layers.Dropout(0.4))
    discriminator_model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    discriminator_model.add(layers.LeakyReLU(alpha=0.2))
    discriminator_model.add(layers.Dropout(0.4))
    discriminator_model.add(layers.Flatten())
    discriminator_model.add(layers.Dense(1, activation='sigmoid'))

    optimization = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    discriminator_model.compile(loss='binary_crossentropy', optimizer=optimization, metrics=['accuracy'])

    discriminator_model.summary()
    plot_model(discriminator_model, to_file='discriminator_model.png', show_shapes=True, show_layer_names=True)

    return discriminator_model


def train_discriminator(model, dataset, episodes=100, batch=256):
    half_batch = int(batch / 2)

    for episode in range(episodes):
        x_real, y_real = generate_real_samples(dataset, half_batch)
        _, real_accuracy = model.train_on_batch(x_real, y_real)
        x_fake, y_fake = generate_fake_samples(half_batch)
        _, fake_accuracy = model.train_on_batch(x_fake, y_fake)
        print(f'Episode: {episode+1}: Real = {real_accuracy*100}   Fake = {fake_accuracy*100}')


def train_gan(gan_model, latent_space_dimensions, episodes=100, batch=256):
    for episode in range(episodes):
        x_gan = generate_latent_points(latent_space_dimensions, batch)
        y_gan = np.ones((batch, 1))
        gan_model.train_on_batch(x_gan, y_gan)


if __name__ == '__main__':
    assignment2()
