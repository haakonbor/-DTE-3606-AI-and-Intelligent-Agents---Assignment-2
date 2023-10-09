import matplotlib.pyplot as plt
from keras import layers, models, optimizers
from keras.utils import plot_model
import numpy as np
from main import N_FEATURES, SEQUENCE_LEN


def plot_accuracy(acc_real, acc_fake, epochs):
    x = range(1, epochs + 1)
    y = acc_real, acc_fake

    fig, ax = plt.subplots(1)
    ax.plot(x, y[0])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Discriminator accuracy')
    ax.plot(x, y[1])
    plt.show()


def generate_latent_points(latent_space_dim, samples):
    points = np.random.randn(latent_space_dim * samples)
    points = points.reshape(samples, latent_space_dim)
    return points


def sequence_discriminator(input_structure):
    discriminator_model = models.Sequential(name="DISCRIMINATOR")

    # Hidden layer 1
    discriminator_model.add(layers.Flatten(input_shape=input_structure))
    discriminator_model.add(layers.Dense(512))
    discriminator_model.add(layers.LeakyReLU(alpha=0.2))

    # Hidden layer 2
    discriminator_model.add(layers.Dense(256))
    discriminator_model.add(layers.LeakyReLU(alpha=0.2))

    discriminator_model.add(layers.Dense(1, activation='sigmoid'))

    optimization = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    discriminator_model.compile(loss='binary_crossentropy', optimizer=optimization, metrics=['accuracy'])

    discriminator_model.summary()
    plot_model(discriminator_model, to_file="seq_discriminator_model.png", show_shapes=True, show_layer_names=True)

    return discriminator_model


def sequence_generator(latent_space_dim):
    generator_model = models.Sequential(name="GENERATOR")

    generator_model.add(layers.Dense(256, input_dim=latent_space_dim))
    generator_model.add(layers.BatchNormalization())
    generator_model.add(layers.LeakyReLU(alpha=0.2))

    generator_model.add(layers.Dense(512))
    generator_model.add(layers.BatchNormalization())
    generator_model.add(layers.LeakyReLU(alpha=0.2))

    generator_model.add(layers.Dense(1024))
    generator_model.add(layers.LeakyReLU(alpha=0.2))

    generator_model.add(layers.Dense(N_FEATURES * SEQUENCE_LEN, activation='tanh'))
    generator_model.add(layers.Reshape((SEQUENCE_LEN, N_FEATURES)))

    generator_model.summary()
    plot_model(generator_model, to_file="seq_generator_model.png", show_shapes=True, show_layer_names=True)

    return generator_model


def generator_conv(latent_space_dim):
    generator_model = models.Sequential(name="GENERATOR_CONV")

    generator_model.add(layers.Dense(N_FEATURES/2 * SEQUENCE_LEN/2 * 1024, input_dim=latent_space_dim))
    generator_model.add(layers.BatchNormalization())
    generator_model.add(layers.LeakyReLU(alpha=0.2))
    generator_model.add(layers.Reshape((int(SEQUENCE_LEN / 4), int(N_FEATURES / 4), 1024)))

    generator_model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same'))
    generator_model.add(layers.BatchNormalization())
    generator_model.add(layers.LeakyReLU(alpha=0.2))

    generator_model.add(layers.Conv2D(1, (5, 5), activation='tanh', padding='same'))

    generator_model.summary()
    plot_model(generator_model, to_file='conv_generator_model.png', show_shapes=True, show_layer_names=True)

    return generator_model


def discriminator_conv(input_structure):
    discriminator_model = models.Sequential(name="DISCRIMINATOR_CONV")
    discriminator_model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same',
                                          input_shape=input_structure))
    discriminator_model.add(layers.LeakyReLU(alpha=0.2))

    discriminator_model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    discriminator_model.add(layers.LeakyReLU(alpha=0.2))

    discriminator_model.add(layers.Flatten())
    discriminator_model.add(layers.Dense(1, activation='sigmoid'))

    optimization = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    discriminator_model.compile(loss='binary_crossentropy', optimizer=optimization, metrics=['accuracy'])

    discriminator_model.summary()
    plot_model(discriminator_model, to_file='discriminator_model.png', show_shapes=True, show_layer_names=True)

    return discriminator_model


def sequence_gan(generator, discriminator):
    discriminator.trainable = False
    gan_model = models.Sequential()
    gan_model.add(generator)
    gan_model.add(discriminator)
    optimization = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    gan_model.compile(loss='binary_crossentropy', optimizer=optimization)

    gan_model.summary()
    plot_model(gan_model, to_file="seq_gan_model.png", show_shapes=True, show_layer_names=True)
    return gan_model


def generate_real_sequence_samples(sequences, samples):
    random_index = np.random.randint(0, sequences.shape[0], samples)
    x = sequences[random_index]
    x = np.expand_dims(x, axis=-1)
    y = np.full((samples, 1), 1)
    return x, y


def generate_fake_sequence_samples(generator, latent_space_dim, samples):
    points = generate_latent_points(latent_space_dim, samples)
    x = generator.predict(points).reshape(samples, SEQUENCE_LEN, N_FEATURES)
    x = np.expand_dims(x, axis=-1)
    y = np.full((samples, 1), 0)
    return x, y


def summarize_performance(episode, generator_model, discriminator_model, sequences, latent_space_dim, samples=100):
    x_real, y_real = generate_real_sequence_samples(sequences, samples)
    x_fake, y_fake = generate_fake_sequence_samples(generator_model, latent_space_dim, samples)

    # Evaluate discriminator model on real samples
    _, accuracy_real = discriminator_model.evaluate(x_real, y_real, verbose=0)

    # Evaluate discriminator model on fake samples
    _, accuracy_fake = discriminator_model.evaluate(x_fake, y_fake, verbose=0)

    print(f'Real samples accuracy: {accuracy_real * 100}%  Fake samples accuracy: {accuracy_fake * 100}%')

    # save_plot(x_fake, episode)
    filename = f'generator_model_{episode + 1}.h5'
    generator_model.save(filename)
    plt.close()

    return accuracy_real, accuracy_fake


def train_seq_gan(generator, discriminator, gan, sequences, latent_space_dim, episodes, batch_size):
    batch_per_episode = int(sequences.shape[0] / batch_size)
    half_batch = int(batch_size / 2)

    accuracy_real = []
    accuracy_fake = []

    for episode in range(episodes):
        for i in range(batch_per_episode):
            x_real, y_real = generate_real_sequence_samples(sequences, half_batch)
            x_fake, y_fake = generate_fake_sequence_samples(generator, latent_space_dim, half_batch)

            x_discriminator, y_discriminator = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
            discriminator_loss, _ = discriminator.train_on_batch(x_discriminator, y_discriminator)

            x_gan = generate_latent_points(latent_space_dim, batch_size)
            y_gan = np.ones((batch_size, 1))
            gan_loss = gan.train_on_batch(x_gan, y_gan)

            print(f'Episode {episode + 1}, {i + 1}/{batch_per_episode}  Discriminator loss: '
                  f'{discriminator_loss}    GAN loss: {gan_loss}')

        epoch_accuracy_real, epoch_accuracy_fake = summarize_performance(
            episode, generator, discriminator, sequences, latent_space_dim)
        accuracy_real.append(epoch_accuracy_real)
        accuracy_fake.append(epoch_accuracy_fake)

    return accuracy_real, accuracy_fake
