import collections

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets.mnist import load_data
from keras import layers, models, optimizers
from keras.utils import plot_model
from tensorflow.python.client import device_lib
import pretty_midi
import os
import pygame
from sklearn import preprocessing
from enum import IntEnum
import warnings

warnings.filterwarnings("error")    # To avoid loading corrupted MIDI files
np.random.seed(0)   # For reproduction of results


class NoteAttribute(IntEnum):
    NUMBER = 0
    VELOCITY = 1
    START_TIME = 2
    DURATION = 3


N_FEATURES = len(NoteAttribute)


def assignment2():
    # --- CLASS EXERCISE ---
    # tutorial()
    # ----------------------

    # --- ASSIGNMENT ---
    # Get path of all MIDI files in directory
    root_directory = "clean_midi"
    filepaths = []
    if os.path.exists(root_directory):
        filepaths = get_all_filepaths(root_directory)
    else:
        print(f"Root directory {root_directory} not valid")

    # Load MIDI files
    n_files = 2
    midi_files = load_midi_files(filepaths, n_files)

    # Create sequences of fixed amount of notes
    sequence_length = 16
    sequences = create_sequences(midi_files, sequence_length)

    # Play example sequence
    # sequence_midi = create_midi_sequence(sequences[0])
    # sequence_filepath = "first_sequence_example.mid"
    # sequence_midi.write(sequence_filepath)
    # play_midi_file(sequence_filepath)

    # Normalize the features
    normalized_notes = normalize_features(sequences)
    normalized_sequences = []
    i = 0
    while i < len(normalized_notes):
        normalized_sequences.append(normalized_notes[i:i+sequence_length])
        i += sequence_length
    normalized_sequences = np.array(normalized_sequences)

    random_samples = generate_real_samples(normalized_sequences, 5)
    print(random_samples)


def get_all_filepaths(directory):
    filepaths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            filepaths.append(filepath)

    return filepaths


def load_midi_files(filepaths, n_files):
    midi_files = []
    current_n_files = 0
    i = 0
    while current_n_files < n_files:
        try:
            midi_files.append(pretty_midi.PrettyMIDI(filepaths[i]))
            print(f"READING FILE: {filepaths[i]}")
            print(f'{((current_n_files + 1) / n_files) * 100}%')
            current_n_files += 1
        except Exception as e:
            print(f'ERROR READING MIDI FILE: {e}')
        i += 1
    return midi_files


def create_midi_sequence(notes):
    midi_sequence = pretty_midi.PrettyMIDI()
    instrument_program = pretty_midi.instrument_name_to_program("Piano")
    instrument = pretty_midi.Instrument(program=instrument_program)
    first_note_start_time = notes[0][NoteAttribute.START_TIME]

    for note in notes:
        current_note = pretty_midi.Note(velocity=int(note[NoteAttribute.VELOCITY]),
                                        pitch=int(note[NoteAttribute.NUMBER]),
                                        start=note[NoteAttribute.START_TIME] - first_note_start_time,
                                        end=note[NoteAttribute.START_TIME] - first_note_start_time
                                        + note[NoteAttribute.DURATION])
        instrument.notes.append(current_note)

    midi_sequence.instruments.append(instrument)
    return midi_sequence


def play_midi_file(filepath):
    pygame.mixer.init()
    pygame.mixer.music.load(filepath)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(1000)
    pygame.quit()


def extract_features(file):
    notes = []
    resolution = file.resolution
    minimum_rest_ticks = resolution
    minimum_rest_duration = minimum_rest_ticks / file.estimate_tempo()

    for instrument in file.instruments:
        if instrument.is_drum:
            continue
        prev_note_end = 0

        for note in instrument.notes:
            # Rest note needs to be added due to pause between notes
            pause_duration = note.start - prev_note_end
            if pause_duration > minimum_rest_duration and prev_note_end != 0:
                rest_note = np.zeros(N_FEATURES)
                rest_note[NoteAttribute.NUMBER] = 0
                rest_note[NoteAttribute.VELOCITY] = 0
                rest_note[NoteAttribute.START_TIME] = prev_note_end
                rest_note[NoteAttribute.DURATION] = pause_duration
                notes.append(rest_note)

            current_note = np.zeros(N_FEATURES)
            current_note[NoteAttribute.NUMBER] = note.pitch
            current_note[NoteAttribute.VELOCITY] = note.velocity
            current_note[NoteAttribute.START_TIME] = note.start
            current_note[NoteAttribute.DURATION] = note.end - note.start
            notes.append(current_note)

            prev_note_end = note.end

    return np.array(notes)


def create_sequences(midi_files, sequence_length):
    sequences = []

    for midi in midi_files:
        # Extract features from MIDI file
        midi_notes = extract_features(midi)
        n_notes = len(midi_notes)
        i = 0
        # Only add sequences of continuous notes in same song
        while i < n_notes and n_notes - i > sequence_length:
            sequences.append(midi_notes[i:i + sequence_length])
            i += sequence_length

    return np.array(sequences)


def normalize_features(sequences):
    # Normalize the data
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    note_numbers = []
    normalized_velocities = []
    normalized_start_times = []
    note_durations = []

    for sequence in sequences:
        note_numbers.append(sequence[:, NoteAttribute.NUMBER])

        for note in sequence:
            if note[NoteAttribute.VELOCITY] > 0:
                normalized_velocities.append(1)
            else:
                normalized_velocities.append(0)

        note_start_times = sequence[:, NoteAttribute.START_TIME]
        normalized_start_times.append(scaler.fit_transform(note_start_times.reshape(-1, 1)).ravel())
        note_durations.append(sequence[:, NoteAttribute.DURATION])

    normalized_note_numbers = scaler.fit_transform(np.array(note_numbers).reshape(-1, 1)).ravel()
    normalized_start_times = np.array(normalized_start_times).ravel()
    normalized_durations = scaler.fit_transform(np.array(note_durations).reshape(-1, 1)).ravel()

    normalized_notes = np.column_stack((normalized_note_numbers, normalized_velocities,
                                        normalized_start_times, normalized_durations))
    return normalized_notes


# --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- CLASS EXERCISE -----------------------------------------------
# --------------------------------------------------------------------------------------------------------------
def tutorial():
    print(check_gpu())
    # Hyperparameters
    latent_space_dim = 100
    episodes = 100
    batch = 256

    # Discriminator
    discriminator_model = create_discriminator()
    dataset = load_real_samples()
    # train_discriminator(discriminator_model, dataset)

    # Generator
    generator_model = create_generator(latent_space_dim)

    # GAN
    gan_model = create_gan(generator_model, discriminator_model)

    # Training
    train(generator_model, discriminator_model, gan_model, dataset, latent_space_dim, episodes, batch)

    plt.show()


def check_gpu():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def summarize_performance(episode, generator_model, discriminator_model, dataset, latent_space_dim, samples=100):
    x_real, y_real = generate_real_samples(dataset, samples)
    x_fake, y_fake = generate_fake_samples(generator_model, latent_space_dim, samples)

    # Evaluate discriminator model on real samples
    _, accuracy_real = discriminator_model.evaluate(x_real, y_real, verbose=0)

    # Evaluate discriminator model on fake samples
    _, accuracy_fake = discriminator_model.evaluate(x_fake, y_fake, verbose=0)

    print(f'Real samples accuracy: {accuracy_real * 100}%  Fake samples accuracy: {accuracy_fake * 100}%')

    save_plot(x_fake, episode)
    filename = f'generator_model_{episode + 1}.h5'
    generator_model.save(filename)
    plt.close()


def save_plot(examples, episode, n=10):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    filename = f'generated_plot_episode_{episode + 1}.png'
    plt.savefig(filename)
    plt.close()


def load_real_samples():
    (train_x, _), (_, _) = load_data()
    x = np.expand_dims(train_x, axis=-1)  # Expand from 2D to 3D
    x = x.astype('float32')
    x = x / 255.0  # Normalize data to [0, 1]
    return x


def generate_real_samples(dataset, samples):
    random_index = np.random.randint(0, dataset.shape[0], samples)
    x = dataset[random_index]
    y = np.ones((samples, 1))
    return x, y


def generate_fake_samples(generator_model, latent_space_dim, samples):
    points = generate_latent_points(latent_space_dim, samples)
    x = generator_model.predict(points)
    y = np.zeros((samples, 1))
    return x, y


def generate_latent_points(latent_space_dim, samples):
    points = np.random.randn(latent_space_dim * samples)
    points = points.reshape(samples, latent_space_dim)
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
        print(f'Episode: {episode + 1}: Real = {real_accuracy * 100}   Fake = {fake_accuracy * 100}')


def train_gan(gan_model, latent_space_dim, episodes=100, batch=256):
    for episode in range(episodes):
        x_gan = generate_latent_points(latent_space_dim, batch)
        y_gan = np.ones((batch, 1))
        gan_model.train_on_batch(x_gan, y_gan)


def train(generator_model, discriminator_model, gan_model, dataset, latent_space_dim, episodes=100, batch=256):
    batch_per_episode = int(dataset.shape[0] / batch)
    half_batch = int(batch / 2)

    for episode in range(episodes):
        for i in range(batch_per_episode):
            x_real, y_real = generate_real_samples(dataset, half_batch)
            x_fake, y_fake = generate_fake_samples(generator_model, latent_space_dim, half_batch)

            x_discriminator, y_discriminator = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
            discriminator_loss, _ = discriminator_model.train_on_batch(x_discriminator, y_discriminator)

            x_gan = generate_latent_points(latent_space_dim, batch)
            y_gan = np.ones((batch, 1))
            gan_loss = gan_model.train_on_batch(x_gan, y_gan)

            print(f'Episode {episode + 1}, {i}/{batch_per_episode}  Discriminator loss: '
                  f'{discriminator_loss}    GAN loss: {gan_loss}')

        if (episode + 1) % 10 == 0:
            summarize_performance(episode, generator_model, discriminator_model, dataset, latent_space_dim)


if __name__ == '__main__':
    assignment2()
