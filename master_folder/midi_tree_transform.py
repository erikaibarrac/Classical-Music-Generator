from midi_transform import MidiTransform
import numpy as np

class MidiTreeTransform(MidiTransform):
    def __init__(self, speed=120):
        super(MidiTreeTransform, self).__init__()

        # Add this extra, special speed that will be used with all of the notes if it doesn't yet exist (We don't care about speed here)
        compressed_speed = np.array([speed], dtype=int)
        if compressed_speed.tobytes() in self.speeds_to_numbers:
            self.special_speed = self.speeds_to_numbers[compressed_speed.tobytes()]
        else:
            self.speeds_to_numbers[compressed_speed.tobytes()] = self.speed_dict_length
            self.numbers_to_speeds[self.speed_dict_length] = compressed_speed
            self.special_speed = self.speed_dict_length
            self.speed_dict_length += 1

    def read_midi_files(self, files):
        states_array = []
        notes_array = []
        lengths = []

        for file in files:
            states, notes = self.read_midi_file(file)
            states_array.append(states)
            notes_array.append(notes)
            lengths.append(len(notes))

        return np.concatenate(states_array, axis=0), np.concatenate(notes_array), np.array(lengths)

    def read_midi_file(self, filename):
        """ from a given midi file, return the note and speed number arrays to train with """

        time_steps, notes, speeds = self.get_data(filename)
        master_array = self.generate_master_array(time_steps, notes, speeds)

        notes = self.convert_notes_to_numbers(master_array)

        # Get the state arrays for each time step
        states = []
        curr_state = np.zeros(master_array.shape[1], dtype=int) # Start out with nothing being played

        for note in notes:
            states.append(curr_state)
            curr_state = self.update_state_array(curr_state, note)

        return np.array(states), notes

    def create_state_array(self, notes):
        # Get the state arrays for each time step
        states = []
        curr_state = np.zeros(88, dtype=int) # Start out with nothing being played

        for note in notes:
            states.append(curr_state)
            curr_state = self.update_state_array(curr_state, note)

        return np.array(states, dtype=int)

    def update_state_array(self, state_array, note_number):
        """Takes a state array and a note number and updates the state array to reflect the changes imposed by the note number."""
        note_update = self.numbers_to_notes[note_number]

        state_array = state_array.copy()
        state_array[(state_array > 0)] += 1 # Continue playing these notes
        state_array[(state_array <= 0)] -= 1 # Continue playing these notes
        state_array[(state_array > 0) & (note_update == -1)] = 0 # Stop playing these notes
        state_array[(state_array <= 0) & (note_update == 1)] = 1 # Start playing these notes

        return state_array

    def save_as_midi(self, note_numbers, output_filename):
        """Takes the note numbers and an output file name and saves the note numbers as a midi file"""
        speed_numbers = np.full_like(note_numbers, self.special_speed)
        super(MidiTreeTransform, self).save_as_midi(note_numbers, speed_numbers, output_filename)
