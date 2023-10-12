import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import muspy
import os
import pygame
from enum import IntEnum
import warnings
from GAN import *


class NoteAttribute(IntEnum):
    NUMBER = 0
    VELOCITY = 1


N_FILES = 10
N_FEATURES = len(NoteAttribute)
SEQUENCE_LEN = 16
LATENT_SPACE_DIMENSIONS = 100

OLD_MIN = 0
OLD_MAX = 127

SCALER_MIN = -1
SCALER_MAX = 1

MODEL_TYPE = "DENSE"

np.random.seed(0)   # For reproduction of results


def assignment2():
    warnings.filterwarnings("error")  # To avoid loading corrupted MIDI files
    normalized_sequences = load_and_preprocess_data()
    warnings.filterwarnings("default")

    # example_sequence = create_midi_sequence(normalized_sequences[0])
    # firs_sequence_midi_path = "first_sequence.mid"
    # example_sequence.write(firs_sequence_midi_path)
    # play_midi_file(firs_sequence_midi_path)

    if MODEL_TYPE == "DENSE":
        # Dense layers
        discriminator = sequence_discriminator((SEQUENCE_LEN, N_FEATURES))
        generator = sequence_generator(LATENT_SPACE_DIMENSIONS)

    else:
        # CNN
        discriminator = discriminator_conv((SEQUENCE_LEN, N_FEATURES, 1))
        generator = generator_conv(LATENT_SPACE_DIMENSIONS)

    # GAN model
    gan = sequence_gan(generator, discriminator)

    n_epochs = 100
    batch_size = 128
    accuracy_real, accuracy_fake = train_seq_gan(
        generator, discriminator, gan, normalized_sequences, LATENT_SPACE_DIMENSIONS, n_epochs, batch_size)

    plot_accuracy(accuracy_real, accuracy_fake, n_epochs)

    n_generated_samples = 5
    epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for epoch in epochs:
        generate_new_sequences(n_generated_samples, epoch)

    sequence_statistics(n_generated_samples, epochs)


def sequence_statistics(n, epoch_set):
    example_sequence_score = muspy.read_midi("first_sequence_example.mid")
    example_sequence_pitches = muspy.n_pitches_used(example_sequence_score)
    example_sequence_pitch_range = muspy.pitch_range(example_sequence_score)
    example_sequence_pitch_entropy = muspy.pitch_entropy(example_sequence_score)

    print(
        f"Example sequence:"
        f"number of pitches: {example_sequence_pitches}, pitch range: {example_sequence_pitch_range}, "
        f"pitch entropy: {example_sequence_pitch_entropy} \n")

    generated_sequences_pitches = np.zeros((len(epoch_set), n))
    generated_sequences_pitch_ranges = np.zeros((len(epoch_set), n))
    generated_sequences_pitch_entropies = np.zeros((len(epoch_set), n))

    for i, epoch in enumerate(epoch_set):
        for j in range(n):
            generated_sequence_score = muspy.read_midi(
                f"generated_midi_{SEQUENCE_LEN}/{MODEL_TYPE}/model_{epoch}_generated_sequence_{j}.mid")
            generated_sequences_pitches[i][j] = muspy.n_pitches_used(generated_sequence_score)
            generated_sequences_pitch_ranges[i][j] = muspy.pitch_range(generated_sequence_score)
            generated_sequences_pitch_entropies[i][j] = muspy.pitch_entropy(generated_sequence_score)

    average_pitches = [np.mean(episode) for episode in generated_sequences_pitches]
    average_pitch_ranges = [np.mean(episode) for episode in generated_sequences_pitch_ranges]
    average_pitch_entropies = [np.mean(episode) for episode in generated_sequences_pitch_entropies]

    fig, ax = plt.subplots()
    ax.plot(epoch_set, generated_sequences_pitches, label="Generated sequence")
    ax.plot(epoch_set, average_pitches, linestyle=':', label="Average generated sequence")
    ax.axhline(y=example_sequence_pitches, linestyle='--', label="Example sequence")
    plt.xlabel("Episode")
    plt.ylabel("Number of notes")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(epoch_set, generated_sequences_pitch_ranges, label="Generated sequence")
    ax.plot(epoch_set, average_pitch_ranges, linestyle=':', label="Average generated sequence")
    ax.axhline(y=example_sequence_pitch_range, linestyle='--', label="Example sequence")
    plt.xlabel("Episode")
    plt.ylabel("Pitch range")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(epoch_set, generated_sequences_pitch_entropies, label="Generated sequence")
    ax.plot(epoch_set, average_pitch_entropies, linestyle=':', label="Average generated sequence")
    ax.axhline(y=example_sequence_pitch_entropy, linestyle='--', label="Example sequence")
    plt.xlabel("Episode")
    plt.ylabel("Pitch entropy")
    plt.show()


def generate_new_sequences(n, model_epoch):
    latent_points = generate_latent_points(LATENT_SPACE_DIMENSIONS, n)
    generator = models.load_model(f'GAN_models_{SEQUENCE_LEN}/{MODEL_TYPE}/generator_model_{model_epoch}.h5')
    x = np.squeeze(generator.predict(latent_points))

    for i, generated_sequence in enumerate(x):
        sequence = []
        for note in generated_sequence:
            sequence.append([int(un_normalize(note[NoteAttribute.NUMBER])),
                             int(un_normalize(note[NoteAttribute.VELOCITY]))])
        sequence_midi = create_midi_sequence(sequence)
        sequence_filepath = f"generated_midi_{SEQUENCE_LEN}/{MODEL_TYPE}/model_{model_epoch}_generated_sequence_{i}.mid"
        sequence_midi.write(sequence_filepath)
        # play_midi_file(sequence_filepath)


def load_and_preprocess_data():
    data_filename = f'sequence_data_{SEQUENCE_LEN}/simple_sequence_data.npy'

    if os.path.exists(data_filename):
        sequences = np.load(data_filename)

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

        # Define sequences of fixed amount of notes
        sequences = define_sequences(midi_files, SEQUENCE_LEN)

        # Store the normalized sequences in a file
        np.save(data_filename, sequences)

    normalized_sequences = normalize_sequences(sequences)

    return normalized_sequences


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


def define_sequences(midi_files, sequence_length):
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


def normalize_sequences(sequences):
    normalized_sequences = np.copy(sequences)
    for i, seq in enumerate(sequences):
        for j, note in enumerate(seq):
            normalized_sequences[i][j][NoteAttribute.NUMBER] = normalize(note[NoteAttribute.NUMBER])
            normalized_sequences[i][j][NoteAttribute.VELOCITY] = normalize(note[NoteAttribute.VELOCITY])

    return normalized_sequences


def create_midi_sequence(notes):
    midi_sequence = pretty_midi.PrettyMIDI()
    instrument_program = pretty_midi.instrument_name_to_program("Electric Piano 1")
    instrument = pretty_midi.Instrument(program=instrument_program)
    current_time = 0
    step = 0.5

    for note in notes:
        current_note = pretty_midi.Note(velocity=note[NoteAttribute.VELOCITY],
                                        pitch=note[NoteAttribute.NUMBER],
                                        start=current_time,
                                        end=current_time + step)
        current_time += step
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
    notes = []
    for instrument in file.instruments:
        if instrument.is_drum or len(instrument.notes) < sequence_length:
            continue
        instrument_notes = []

        for note in instrument.notes:
            current_note = np.zeros(N_FEATURES)
            current_note[NoteAttribute.NUMBER] = note.pitch
            current_note[NoteAttribute.VELOCITY] = note.velocity

            instrument_notes.append(current_note)

        notes.append(instrument_notes)

    return notes


def normalize(value):
    return ((value - OLD_MIN) / (OLD_MAX - OLD_MIN)) * (SCALER_MAX - SCALER_MIN) + SCALER_MIN


def un_normalize(scaled_value):
    return ((scaled_value - SCALER_MIN) / (SCALER_MAX - SCALER_MIN)) * (OLD_MAX - OLD_MIN) + OLD_MIN


if __name__ == '__main__':
    assignment2()
