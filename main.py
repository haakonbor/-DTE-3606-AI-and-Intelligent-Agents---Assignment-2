import pretty_midi
import muspy
import os
import pygame
from enum import IntEnum
import warnings
from GAN import *


class NoteAttribute(IntEnum):
    NUMBER = 0
    DURATION = 1


N_FILES = 10
N_FEATURES = len(NoteAttribute)
SEQUENCE_LEN = 16
LATENT_SPACE_DIMENSIONS = 100

MIN_PITCH = 0
MAX_PITCH = 127
MIN_DURATION = 0
MAX_DURATION = 0
SCALER_MIN = -1
SCALER_MAX = 1

np.random.seed(0)   # For reproduction of results


def assignment2():
    warnings.filterwarnings("error")  # To avoid loading corrupted MIDI files
    normalized_sequences = load_and_preprocess_data()
    warnings.filterwarnings("default")

    # example_sequence = create_midi_sequence(normalized_sequences[0])
    # firs_sequence_midi_path = "first_sequence.mid"
    # example_sequence.write(firs_sequence_midi_path)
    # play_midi_file(firs_sequence_midi_path)

    # Dense layers
    # discriminator = sequence_discriminator((SEQUENCE_LEN, N_FEATURES))
    # generator = sequence_generator(LATENT_SPACE_DIMENSIONS)

    # CNN
    discriminator = discriminator_conv((SEQUENCE_LEN, N_FEATURES, 1))
    generator = generator_conv(LATENT_SPACE_DIMENSIONS)

    # GAN model
    gan = sequence_gan(generator, discriminator)

    n_epochs = 100
    batch_size = 128
    # accuracy_real, accuracy_fake = train_seq_gan(
    #     generator, discriminator, gan, normalized_sequences, LATENT_SPACE_DIMENSIONS, n_epochs, batch_size)

    # plot_accuracy(accuracy_real, accuracy_fake, n_epochs)

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

    for epoch in epoch_set:
        generated_sequences_pitches = np.zeros(n)
        generated_sequences_pitch_ranges = np.zeros(n)
        generated_sequences_pitch_entropies = np.zeros(n)
        for i in range(n):
            generated_sequence_score = muspy.read_midi(
                f"generated_midi_{SEQUENCE_LEN}/model_{epoch}_generated_sequence_{i}.mid")
            generated_sequences_pitches[i] = muspy.n_pitches_used(generated_sequence_score)
            generated_sequences_pitch_ranges[i] = muspy.pitch_range(generated_sequence_score)
            generated_sequences_pitch_entropies[i] = muspy.pitch_entropy(generated_sequence_score)

        print(
            f"GAN model at epoch {epoch} generated sequences average:"
            f"number of pitches: {np.mean(generated_sequences_pitches)}, "
            f"pitch range: {np.mean(generated_sequences_pitch_ranges)}, "
            f"pitch entropy: {np.mean(generated_sequences_pitch_entropies)} \n")


def generate_new_sequences(n, model_epoch):
    latent_points = generate_latent_points(LATENT_SPACE_DIMENSIONS, n)
    generator = models.load_model(f'GAN_models_{SEQUENCE_LEN}/generator_model_{model_epoch}.h5')
    x = np.squeeze(generator.predict(latent_points))

    for i, generated_sequence in enumerate(x):
        sequence = []
        for note in generated_sequence:
            sequence.append([int(un_normalize_pitch(note[NoteAttribute.NUMBER])), un_normalize_duration(note[NoteAttribute.DURATION])])
        sequence_midi = create_midi_sequence(sequence)
        sequence_filepath = f"generated_midi_{SEQUENCE_LEN}/model_{model_epoch}_generated_sequence_{i}.mid"
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
    global MIN_DURATION
    global MAX_DURATION

    durations = np.zeros(SEQUENCE_LEN * len(sequences))
    n = 0

    for seq in sequences:
        for note in seq:
            durations[n] = note[NoteAttribute.DURATION]
            n += 1

    MIN_DURATION = np.min(durations)
    MAX_DURATION = np.max(durations)

    normalized_sequences = np.copy(sequences)
    for i, seq in enumerate(sequences):
        for j, note in enumerate(seq):
            normalized_sequences[i][j][NoteAttribute.NUMBER] = normalize_pitch(note[NoteAttribute.NUMBER])
            normalized_sequences[i][j][NoteAttribute.DURATION] = normalize_duration(note[NoteAttribute.DURATION])

    return normalized_sequences


def create_midi_sequence(notes):
    midi_sequence = pretty_midi.PrettyMIDI()
    instrument_program = pretty_midi.instrument_name_to_program("Electric Piano 1")
    instrument = pretty_midi.Instrument(program=instrument_program)
    current_time = 0
    # step = 0.5

    for note in notes:
        current_note = pretty_midi.Note(velocity=80,
                                        pitch=note[NoteAttribute.NUMBER],
                                        start=current_time,
                                        end=current_time + note[NoteAttribute.DURATION])
        current_time += note[NoteAttribute.DURATION]
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
            current_note[NoteAttribute.DURATION] = note.end - note.start

            if current_note[NoteAttribute.DURATION] > 5:
                continue

            instrument_notes.append(current_note)

        notes.append(instrument_notes)

    return notes


def normalize_pitch(value):
    return ((value - MIN_PITCH) / (MAX_PITCH - MIN_PITCH)) * (SCALER_MAX - SCALER_MIN) + SCALER_MIN


def un_normalize_pitch(scaled_value):
    return ((scaled_value - SCALER_MIN) / (SCALER_MAX - SCALER_MIN)) * (MAX_PITCH - MIN_PITCH) + MIN_PITCH


def normalize_duration(value):
    return ((value - MIN_DURATION) / (MAX_DURATION - MIN_DURATION)) * (SCALER_MAX - SCALER_MIN) + SCALER_MIN


def un_normalize_duration(scaled_value):
    return ((scaled_value - SCALER_MIN) / (SCALER_MAX - SCALER_MIN)) * (MAX_DURATION - MIN_DURATION) + MIN_DURATION


if __name__ == '__main__':
    assignment2()
