import joblib
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models, optimizers
from keras.utils import plot_model
import pretty_midi
import os
import pygame
from sklearn import preprocessing
from enum import IntEnum
import warnings


np.random.seed(0)   # For reproduction of results


class NoteAttribute(IntEnum):
    NUMBER = 0
    VELOCITY = 1
    START_TIME = 2
    DURATION = 3


N_FILES = 100
N_FEATURES = len(NoteAttribute)
SEQUENCE_LEN = 32
LATENT_SPACE_DIMENSIONS = 100


def assignment2():
    # --- CLASS EXERCISE ---
    # tutorial()
    # ----------------------

    # # --- ASSIGNMENT ---
    warnings.filterwarnings("error")  # To avoid loading corrupted MIDI files
    normalized_sequences, scalers = load_and_preprocess_data()
    warnings.filterwarnings("default")

    # discriminator = sequence_discriminator((SEQUENCE_LEN, N_FEATURES))
    # generator = sequence_generator(LATENT_SPACE_DIMENSIONS)
    discriminator = discriminator_conv((SEQUENCE_LEN, N_FEATURES, 1))
    generator = generator_conv(LATENT_SPACE_DIMENSIONS)
    gan = sequence_gan(generator, discriminator)

    epochs = 100
    batch_size = 512
    accuracy_real, accuracy_fake = train_seq_gan(
        generator, discriminator, gan, normalized_sequences, LATENT_SPACE_DIMENSIONS, epochs, batch_size)

    plot_accuracy(accuracy_real, accuracy_fake, epochs)

    latent_points = generate_latent_points(LATENT_SPACE_DIMENSIONS, 5)
    generator_100_episodes = models.load_model('generator_model_100.h5')
    x = generator_100_episodes.predict(latent_points)

    for i, generated_sequence in enumerate(x):
        sequence = un_normalize_features(generated_sequence, scalers)
        sequence_midi = create_midi_sequence(sequence)
        sequence_filepath = f"generated_sequence_{i}.mid"
        sequence_midi.write(sequence_filepath)


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

    generator_model.add(layers.Dense(N_FEATURES/4 * SEQUENCE_LEN/4 * 1024, input_dim=latent_space_dim))
    generator_model.add(layers.BatchNormalization())
    generator_model.add(layers.LeakyReLU(alpha=0.2))
    generator_model.add(layers.Reshape((int(SEQUENCE_LEN / 4), int(N_FEATURES / 4), 1024)))

    generator_model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same'))
    generator_model.add(layers.BatchNormalization())
    generator_model.add(layers.LeakyReLU(alpha=0.2))

    generator_model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same'))
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
    y = np.full((samples, 1), 0.9)
    return x, y


def generate_fake_sequence_samples(generator, latent_space_dim, samples):
    points = generate_latent_points(latent_space_dim, samples)
    x = generator.predict(points).reshape(samples, SEQUENCE_LEN, N_FEATURES)
    x = np.expand_dims(x, axis=-1)
    y = np.full((samples, 1), 0.1)
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


def load_and_preprocess_data():
    data_filename = 'sequence_data.npy'
    if os.path.exists(data_filename):
        # Load normalized sequences and scalers from files
        scalers = [joblib.load(f'scaler_feature_{i}.joblib') for i in range(N_FEATURES)]
        return np.load(data_filename), scalers

    else:
        # Get path of all MIDI files in directory
        root_directory = "clean_midi"
        filepaths = []
        if os.path.exists(root_directory):
            filepaths = get_all_filepaths(root_directory)
        else:
            print(f"Root directory {root_directory} not valid")

        # Load MIDI files
        midi_files = load_midi_files(filepaths[:N_FILES])

        # Create sequences of fixed amount of notes
        sequences = create_sequences(midi_files, SEQUENCE_LEN)

        # Normalize the features
        normalized_notes, scalers = normalize_features(sequences)
        normalized_sequences = []
        i = 0
        while i < len(normalized_notes):
            normalized_sequences.append(normalized_notes[i:i + SEQUENCE_LEN])
            i += SEQUENCE_LEN
        normalized_sequences = np.array(normalized_sequences)

        # Store the normalized sequences and scalers in files
        np.save(data_filename, normalized_sequences)
        for i, scaler in enumerate(scalers):
            joblib.dump(scaler, f'scaler_feature_{i}.joblib')

        return normalized_sequences, scalers


def get_all_filepaths(directory):
    filepaths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            filepaths.append(filepath)

    return filepaths


def load_midi_files(filepaths):
    midi_files = []
    n_files = len(filepaths)
    i = 0
    for file in filepaths:
        try:
            midi_files.append(pretty_midi.PrettyMIDI(file))
            print(f"READING FILE: {file}")
            print(f'{(((i + 1) / n_files) * 100): .2f} %')
        except Exception as e:
            print(f'ERROR READING MIDI FILE: {e}')
        i += 1
    return midi_files


def create_midi_sequence(notes):
    midi_sequence = pretty_midi.PrettyMIDI()
    instrument_program = pretty_midi.instrument_name_to_program("Electric Piano 1")
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


def extract_features(file, sequence_length):
    try:
        tempo = file.estimate_tempo()
    except ValueError:      # File has less than two notes so can't estimate tempo
        return []

    notes = []
    resolution = file.resolution
    minimum_rest_ticks = resolution
    minimum_rest_duration = minimum_rest_ticks / tempo

    for instrument in file.instruments:
        if instrument.is_drum or len(instrument.notes) < sequence_length:
            continue
        prev_note_end = 0
        instrument_notes = []

        for note in instrument.notes:
            # Rest note needs to be added due to pause between notes
            pause_duration = note.start - prev_note_end
            if pause_duration > minimum_rest_duration and prev_note_end != 0:
                rest_note = np.zeros(N_FEATURES)
                rest_note[NoteAttribute.NUMBER] = 0
                rest_note[NoteAttribute.VELOCITY] = 0
                rest_note[NoteAttribute.START_TIME] = prev_note_end
                rest_note[NoteAttribute.DURATION] = pause_duration

                instrument_notes.append(rest_note)

            current_note = np.zeros(N_FEATURES)
            current_note[NoteAttribute.NUMBER] = note.pitch
            current_note[NoteAttribute.VELOCITY] = note.velocity
            current_note[NoteAttribute.START_TIME] = note.start
            current_note[NoteAttribute.DURATION] = note.end - note.start

            instrument_notes.append(current_note)

            prev_note_end = note.end

        notes.append(instrument_notes)

    return notes


def create_sequences(midi_files, sequence_length):
    sequences = []

    for midi in midi_files:
        # Extract features from MIDI file
        midi_notes = extract_features(midi, sequence_length)
        # Only add sequences of continuous notes of same instrument
        for instrument_notes in midi_notes:
            n_notes = len(instrument_notes)
            i = 0
            while i < n_notes and n_notes - i > sequence_length:
                sequences.append(instrument_notes[i:i + sequence_length])
                i += sequence_length

    return np.array(sequences)


def normalize_features(sequences):
    # Normalize the data
    scalers = [preprocessing.MinMaxScaler(feature_range=(-1, 1)) for _ in range(N_FEATURES)]

    note_numbers = []
    normalized_velocities = []
    note_start_times = []
    note_durations = []
    max_start_time_range = 0

    for sequence in sequences:
        for note in sequence:
            if note[NoteAttribute.VELOCITY] > 0:
                normalized_velocities.append(1)
            else:
                normalized_velocities.append(0)

        note_numbers.append(sequence[:, NoteAttribute.NUMBER])
        note_start_times.append(sequence[:, NoteAttribute.START_TIME])
        note_durations.append(sequence[:, NoteAttribute.DURATION])

        min_start_time = np.min(sequence[:, NoteAttribute.START_TIME])
        max_start_time = np.max(sequence[:, NoteAttribute.START_TIME])
        start_time_range = max_start_time - min_start_time
        if start_time_range > max_start_time_range:
            max_start_time_range = start_time_range

    scalers[NoteAttribute.START_TIME] = preprocessing.MinMaxScaler(feature_range=(0, max_start_time_range))

    normalized_note_numbers = scalers[NoteAttribute.NUMBER].fit_transform(
        np.array(note_numbers).reshape(-1, 1)).ravel()

    normalized_start_times = scalers[NoteAttribute.START_TIME].fit_transform(
        np.array(note_start_times).reshape(-1, 1)).ravel()

    normalized_durations = scalers[NoteAttribute.DURATION].fit_transform(
        np.array(note_durations).reshape(-1, 1)).ravel()

    normalized_notes = np.column_stack((normalized_note_numbers, normalized_velocities,
                                        normalized_start_times, normalized_durations))
    return normalized_notes, scalers


def un_normalize_features(sequence, scalers):
    original_sequence = np.zeros((SEQUENCE_LEN, N_FEATURES))
    for i, note in enumerate(sequence):
        number_scaler = scalers[NoteAttribute.NUMBER]
        note_number = note[NoteAttribute.NUMBER].reshape(-1, 1)
        original_sequence[i][NoteAttribute.NUMBER] = number_scaler.inverse_transform(note_number)

        start_time_scaler = scalers[NoteAttribute.START_TIME]
        original_sequence[i][NoteAttribute.START_TIME] = start_time_scaler.inverse_transform(
            note[NoteAttribute.START_TIME].reshape(-1, 1))

        duration_scaler = scalers[NoteAttribute.DURATION]
        original_sequence[i][NoteAttribute.DURATION] = duration_scaler.inverse_transform(
            note[NoteAttribute.DURATION].reshape(-1, 1))

        if note[NoteAttribute.VELOCITY] > 0:
            original_sequence[i][NoteAttribute.VELOCITY] = 60

    return original_sequence


if __name__ == '__main__':
    assignment2()
