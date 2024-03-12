import py_midicsv as pm  # Documentation at https://github.com/timwedde/py_midicsv
import pandas as pd
import io
import numpy as np

class MidiTransform():

    def __init__(self):
        self.notes_to_numbers = {}
        self.numbers_to_notes = {}
        self.speeds_to_numbers = {}
        self.numbers_to_speeds = {}
        self.note_dict_length = 0 # ;)
        self.speed_dict_length = 0 # ;}

    def get_data(self, filename, track_num=2):
        """ read in midi file and return time_steps, notes, and speeds arrays """

        csv_midi = pm.midi_to_csv(filename)

        df = pd.read_csv(io.StringIO("".join(csv_midi)), names=["track", "time",
            "type", "extra_0", "extra_1", "extra_2", "extra_3", "extra_4"],
            on_bad_lines="warn", skipinitialspace=True)

        filtered = df.loc[df["track"] == track_num, :]
        notes = filtered[filtered["type"] == "Note_on_c"]

        return np.array(notes["time"]).astype(int), np.array(notes["extra_1"]).astype(int), np.array(notes["extra_2"]).astype(int)


    def generate_master_array(self, time_steps, notes, speeds):
        """ Create a master array that shows the state of each piano key at every time step of the song
        The first column of master_array indicates if the key is being pressed (1), depressed (-1), or the same (0)
        The second column indicates the speed the key is being pressed; a 0 indicates that it's not being pressed
        """

        master_array = np.zeros((time_steps[-1] + 1, 88, 2), dtype=int)

        j = 0
        for i in range(len(time_steps)):

            while j != time_steps[i]:
                j += 1

            if speeds[i] > 0:
                """ if the speed is positive, then the note is being pressed, so set it as a 1 """
                master_array[j][int(notes[i]) - 20] = [1, speeds[i]]

            elif speeds[i] == 0:
                """ if the speed is 0, the note is being depressed, so set it as a -1"""
                master_array[j][int(notes[i]) - 20] = [-1, speeds[i]]

        """ now that the master_array is filled in, we will now find all unique note values for any given time step, and
            then we'll assign each unique vector to a number in a dictionary """

        note_columns = master_array[:, :, 0]
        speed_columns = master_array[:, :, 1]
        unique_note_columns = np.unique(note_columns, axis=0)
        unique_speed_columns = np.unique(speed_columns, axis=0)

        for i in range(len(unique_note_columns)):
            byte_array = unique_note_columns[i].tobytes()
            if byte_array not in self.notes_to_numbers:
                self.numbers_to_notes[self.note_dict_length] = unique_note_columns[i]
                self.notes_to_numbers[byte_array] = self.note_dict_length
                self.note_dict_length += 1

        for i in range(len(unique_speed_columns)):
            condensed_speed_column = unique_speed_columns[i][unique_speed_columns[i] != 0]
            byte_array = condensed_speed_column.tobytes()
            if byte_array not in self.speeds_to_numbers:
                self.numbers_to_speeds[self.speed_dict_length] = condensed_speed_column
                self.speeds_to_numbers[byte_array] = self.speed_dict_length
                self.speed_dict_length += 1

        return master_array.astype(int)

    def convert_notes_to_numbers(self, master_array):
        """ this function takes the self.notes_to_numbers dictionary and returns an array of digits where each digit represents
            one of the unique actions that occurs in this song """
        digits_array = []

        for i in range(master_array.shape[0]):
            digits_array.append(self.notes_to_numbers[master_array[i, :, 0].tobytes()])

        return np.array(digits_array, dtype=int)


    def convert_numbers_to_notes(self, note_numbers):
        """ this function takes the numbers_to_arrays dictionary and returns an array of notes """
        note_array = []

        for i in range(note_numbers.shape[0]):
            note_array.append(self.numbers_to_notes[note_numbers[i]])

        return np.array(note_array, dtype=int)

    def convert_speeds_to_numbers(self, master_array):
        """ this function takes the self.speeds_to_numbers dictionary and returns an array of digits where each digit represents
            one of the unique actions that occurs in this song """
        digits_array = []

        for i in range(master_array.shape[0]):
            digits_array.append(self.speeds_to_numbers[master_array[i, :, 1][master_array[i, :, 1] != 0].tobytes()])

        return np.array(digits_array, dtype=int)

    def convert_numbers_to_speeds(self, speed_numbers, new_notes):
        """ This function converts speed numbers to a speed array given the notes.
        It assumes that the notes and speed assignments are consistent! """
        speed_array = []

        for i in range(speed_numbers.shape[0]):
            mask = new_notes[i] == 1
            decoded = self.numbers_to_speeds[speed_numbers[i]]

            speed_column = np.zeros_like(new_notes[i], dtype=int)
            speed_column[mask] = decoded

            speed_array.append(speed_column)

        return np.array(speed_array, dtype=int)

    def read_midi_file(self, files):
        """ FILES IS AN ARRAY OF MUSIC FILES!!! """
        """ from a given midi file, return the note and speed number arrays to train with """

        time_steps, notes, speeds = self.get_data(files)
        master_array = self.generate_master_array(time_steps, notes, speeds)

        # TO DO: find out what we did to get notes and speeds arrays
        notes = self.convert_notes_to_numbers(master_array)
        speeds = self.convert_speeds_to_numbers(master_array)

        return notes, speeds


    def make_consistent(self, note_numbers, speed_numbers):
        """ This function makes note numbers and speed numbers consistent.
            I.e. checks to make sure that notes are only pressed after they have been depressed
            and that single notes do not have multiple speeds assigned to them.
            Returns two arrays notes and speeds, respectively."""

        notes = self.convert_numbers_to_notes(note_numbers)
        curr_note_states = np.zeros(notes.shape[1], dtype=int) # This is good

        speeds_array = []

        for i in range(notes.shape[0]):
            # Check note consistency
            curr_note_states += notes[i]

            # A note was depressed that was not being pressed
            notes[i][curr_note_states < 0] = 0 # Don't depress it

            # A note was pressed that is already being pressed
            notes[i][curr_note_states > 1] = 0 # Don't press it

            curr_note_states[curr_note_states < 0] = 0 # Note is not being pressed
            curr_note_states[curr_note_states > 1] = 1 # Note is currently being pressed

            # Check speed consistency
            # To do this, check that there are exactly the number of speeds as there are notes.
            speed = np.zeros_like(notes[i], dtype=int)

            num_speeds_needed = np.sum(notes[i] == 1)
            speeds_given = self.numbers_to_speeds[speed_numbers[i]]
            num_speeds_given = len(speeds_given)
            # There are three cases:
            if num_speeds_given == num_speeds_needed:
                speed[notes[i] == 1] = speeds_given
            #elif num_speeds_given < num_speeds_needed:
            else:
                speed[notes[i] == 1] = int(np.mean(speeds_given)) if len(speeds_given) > 0 else 0

            speeds_array.append(speed)

        return notes, np.array(speeds_array, dtype=int)

    def save_as_midi(self, note_numbers, speed_numbers, output_filename):
        """ convert notes and speeds arrays into a functional midi file """
        # Make the notes and speeds consistent
        notes, speeds = self.make_consistent(note_numbers, speed_numbers)

        # Add header and footer information
        header_info = [
            (0, 0, "Header", 1, 2, 240, pd.NA, pd.NA),
            (1, 0, "Start_track", pd.NA, pd.NA, pd.NA, pd.NA, pd.NA),
            (1, 0, "Time_signature", 4, 2, 24, 8, pd.NA),
            (1, 0, "End_track", pd.NA, pd.NA, pd.NA, pd.NA, pd.NA)
        ]
        footer_info = [
            (0, 0, "End_of_file", pd.NA, pd.NA, pd.NA, pd.NA, pd.NA)
        ]

        # Convert sample_notes array into data frame
        piano_track = 2 # This seems typical
        piano_channel = 0 # We've never seen this change
        note_offset = 20 # This is from our conversion from csv to array

        data = []
        data.append((2, 0, "Start_track", pd.NA, pd.NA, pd.NA, pd.NA, pd.NA))
        for time in range(notes.shape[0]): # This tells the time
            for index in range(notes.shape[1]): # This tells which note (-20)
                value = notes[time,index]
                if value != 0:
                    note = index + note_offset
                    velocity = 0 if value == -1 else speeds[time, index] # 0 means turn a note off, otherwise look at speeds to determine note velocity
                    # Rows look like: track, time, type, channel, note, velocity
                    data.append((int(piano_track), int(time), "Note_on_c", int(piano_channel), int(note), int(velocity), pd.NA, pd.NA))
        data.append((2, len(notes), "End_track", pd.NA, pd.NA, pd.NA, pd.NA, pd.NA))

        notes_df = pd.DataFrame(header_info + data + footer_info, columns=["track", "time", "type", "extra_0", "extra_1", "extra_2", "extra_3", "extra_4"])

        out_buf1 = io.StringIO()
        out_buf2 = io.StringIO()
        notes_df.to_csv(out_buf1, header=False, index=False)
        out_buf1.seek(0)

        stripped_lines = []
        for line in out_buf1.readlines():
            stripped_lines.append(line.strip().strip(",")+"\n")

        out_buf2.writelines(stripped_lines)
        out_buf2.seek(0)

        new_song_midi = pm.csv_to_midi(out_buf2)

        with open(output_filename, "wb") as outfile:
            writer = pm.FileWriter(outfile)
            writer.write(new_song_midi)
        return
