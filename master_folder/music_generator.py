#!/usr/bin/env python
# coding: utf-8

# In[1]:


import py_midicsv as pm  # Documentation at https://github.com/timwedde/py_midicsv
import pandas as pd
import io
import numpy as np
from hmmlearn import hmm
# from midi2audio import FluidSynth
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from midi_tree_transform import MidiTreeTransform
import os
from keys_and_times import GetKeys, GetTimes, KeyFilter, TimeFilter


# In[3]:


song_list = os.listdir('Study_Music')
song_list = ["Study_Music/" + song_list[i] for i in range(len(song_list))]

print("Start reading in music")

transformer = MidiTreeTransform()
states, notes, lengths = transformer.read_midi_files(song_list)

print("Finished constructing Transformer")
# In[ ]:


time_filter = TimeFilter(transformer)
compressed_notes = time_filter.note_compress(notes)
compressed_states = transformer.create_state_array(compressed_notes)

key_class = GetKeys()
key_name = key_class.random_key(song_list)
sample_key = key_class.get_notes_from_key(key_name)

key_filter = KeyFilter(sample_key, transformer)
states, notes = key_filter.filter(compressed_states, compressed_notes)

print("Filtered Successfully")

# In[ ]:


# times = GetTimes()
# song_length = times.get_sample_length(song_list)
song_length = 120000

# In[ ]:

print("Start training RF")

rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=False, min_samples_split=100)
rf.fit(states, notes)

print("RF fitted")

curr_state = np.zeros(states.shape[1], dtype=int)

generated_notes = []
for i in range(song_length):
    p = rf.predict_proba(curr_state.reshape(1,-1))
    p = p[0]

    # Throttle the probability of nothing
    p[0] *= .8
    the_rest = p[1:]

    rem_prop = 1. - p[0]
    scale_fact = rem_prop/np.sum(the_rest)

    p[1:] *= scale_fact

    gen_note = np.random.choice(key_filter.filtered_dict_length, p=p)

    generated_notes.append(gen_note)

    curr_state = key_filter.update_state_array(curr_state, gen_note)

print("RF Song Generated")

generated_notes = np.array(generated_notes)

generated_notes = key_filter.sift(generated_notes)
generated_notes = time_filter.enjoy_the_silence(generated_notes)
generated_notes

transformer.save_as_midi(generated_notes, "final_forest_song.midi")

print("RF song saved")

# In[3]:

print("HMM model started")

notes = hmm.MultinomialHMM(n_components=20, n_iter=100, tol=0.01, verbose=False)
notes.fit(notes.reshape(-1, 1), lengths=lengths)

print("HMM model fitted")

new_notes = notes.sample(song_length)

generated_notes = np.array(new_notes)

print("HMM Song Generated")

generated_notes = key_filter.sift(generated_notes)
generated_notes = time_filter.enjoy_the_silence(generated_notes)
generated_notes

transformer.save_as_midi(generated_notes, "final_hmm_song.midi")

print("HMM song saved")
