import pretty_midi
import os
import pygame
from enum import IntEnum
import warnings
from GAN import *


class NoteAttribute(IntEnum):
    NUMBER = 0
    VELOCITY = 1


N_FILES = 100
N_FEATURES = len(NoteAttribute)
SEQUENCE_LEN = 32
LATENT_SPACE_DIMENSIONS = 100

np.random.seed(0)   # For reproduction of results


def assignment2():
    warnings.filterwarnings("error")  # To avoid loading corrupted MIDI files
    normalized_sequences = load_and_preprocess_data()
    warnings.filterwarnings("default")

    example_sequence = create_midi_sequence(normalized_sequences[0])
    firs_sequence_midi_path = "first_sequence.mid"
    example_sequence.write(firs_sequence_midi_path)
    play_midi_file(firs_sequence_midi_path)

    # Dense layers
    discriminator = sequence_discriminator((SEQUENCE_LEN, N_FEATURES))
    generator = sequence_generator(LATENT_SPACE_DIMENSIONS)

    # CNN
    # discriminator = discriminator_conv((SEQUENCE_LEN, N_FEATURES, 1))
    # generator = generator_conv(LATENT_SPACE_DIMENSIONS)

    # GAN model
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
        sequence = []
        for note in generated_sequence:
            sequence.append([un_normalize(note[NoteAttribute.NUMBER]), un_normalize(note[NoteAttribute.VELOCITY])])
        sequence_midi = create_midi_sequence(sequence)
        sequence_filepath = f"generated_sequence_{i}.mid"
        sequence_midi.write(sequence_filepath)


def load_and_preprocess_data():
    data_filename = 'simple_sequence_data.npy'
    if os.path.exists(data_filename):
        return np.load(data_filename)
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

        # Create sequences of fixed amount of notes with normalized values
        sequences = create_sequences(midi_files, SEQUENCE_LEN)

        # Store the normalized sequences in a file
        np.save(data_filename, sequences)

        return sequences


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
    current_time = 0
    step = 0.5

    for note in notes:
        current_note = pretty_midi.Note(velocity=int(un_normalize(note[NoteAttribute.VELOCITY])),
                                        pitch=int(un_normalize(note[NoteAttribute.NUMBER])),
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
            current_note[NoteAttribute.NUMBER] = normalize(note.pitch)
            current_note[NoteAttribute.VELOCITY] = normalize(note.velocity)
            instrument_notes.append(current_note)

        notes.append(instrument_notes)

    return notes


def normalize(value):
    old_min = 0
    old_max = 127
    new_min = -1
    new_max = 1

    return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


def un_normalize(scaled_value):
    old_min = 0
    old_max = 127
    new_min = -1
    new_max = 1

    return ((scaled_value - new_min) / (new_max - new_min)) * (old_max - old_min) + old_min


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


if __name__ == '__main__':
    assignment2()
