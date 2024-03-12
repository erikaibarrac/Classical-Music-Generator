import py_midicsv as pm  # Documentation at https://github.com/timwedde/py_midicsv
import pandas as pd
import io
import numpy as np
import midi_transform as mt

class OptimalTime():
    # 24 Midi Clock Times per quarter note
    # Tempo is in micro seconds 

    def __init__(self, filename = None, track_num = 2):
        self.midi = mt.MidiTransform()
        self.time_steps, self.notes, self.speeds = self.midi.get_data(filename, track_num)

        csv_midi = pm.midi_to_csv(filename)
        df = pd.read_csv(io.StringIO("".join(csv_midi)), names=["track", "time",
            "type", "extra_0", "extra_1", "extra_2", "extra_3", "extra_4"],
            on_bad_lines="warn", skipinitialspace=True)

        self.tempo = int(df.iloc[2, :][3])
        self.thirty_second_note_mtc = 24/32