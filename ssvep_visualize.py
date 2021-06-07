# Some standard pythonic imports
import os, numpy as np, pandas as pd
from collections import OrderedDict
import warnings

warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt

# MNE functions
from mne import Epochs, find_events
from mne.time_frequency import psd_welch, tfr_morlet

# EEG-Notebooks functions
from eegnb.analysis.utils import load_data, plot_conditions
from eegnb.datasets import fetch_dataset

# sphinx_gallery_thumbnail_number = 3

chs = ['O1', 'O2', 'P3', 'Pz', 'P4']

board_name = "cyton"
experiment = "visual_ssvep"
subject_id = 0
session_nb = 0
raw = load_data(subject_id, session_nb,
                experiment=experiment, site='local', device_name=board_name,
                data_dir='data',
                replace_ch_names={'Fp1': 'O1',
                                  'Fp2': 'O2',
                                  'C3': 'P3',
                                  'C4': 'Pz',
                                  'P7': 'P4',
                                  'P8': 'F1',
                                  'O1': 'F2',
                                  'O2': 'F3'})

raw.plot_psd(picks=['O1', 'O2', 'P3', 'Pz', 'P4'])
events = find_events(raw)
event_id = {'30 Hz': 1, '20 Hz': 2}
epochs = Epochs(raw, events=events, event_id=event_id,
                tmin=-0.5, tmax=4, baseline=None, preload=True,
                verbose=False, picks=[0, 1, 2, 3, 4])
print('sample drop %: ', (1 - len(epochs.events) / len(events)) * 100)

f, axs = plt.subplots(2, 1, figsize=(10, 10))
psd1, freq1 = psd_welch(epochs['30 Hz'], n_fft=1028, n_per_seg=256 * 3, picks='all')
psd2, freq2 = psd_welch(epochs['20 Hz'], n_fft=1028, n_per_seg=256 * 3, picks='all')
psd1 = 10 * np.log10(psd1)
psd2 = 10 * np.log10(psd2)

psd1_mean = psd1.mean(0)
psd1_std = psd1.mean(0)

psd2_mean = psd2.mean(0)
psd2_std = psd2.mean(0)

axs[0].plot(freq1, psd1_mean[[0, 3], :].mean(0), color='b', label='30 Hz')
axs[0].plot(freq2, psd2_mean[[0, 3], :].mean(0), color='r', label='20 Hz')

axs[1].plot(freq1, psd1_mean[4, :], color='b', label='30 Hz')
axs[1].plot(freq2, psd2_mean[4, :], color='r', label='20 Hz')

axs[0].set_title('O1 and Pz')
axs[1].set_title('P4')

axs[0].set_ylabel('Power Spectral Density (dB)')
axs[1].set_ylabel('Power Spectral Density (dB)')

axs[0].set_xlim((2, 50))
axs[1].set_xlim((2, 50))

axs[1].set_xlabel('Frequency (Hz)')

axs[0].legend()
axs[1].legend()

plt.show()

# With this visualization we can clearly see distinct peaks at 30hz and 20hz in the PSD, corresponding to the frequency of the visual stimulation. The peaks are much larger at the POz electrode, but still visible at TP9 and TP10

# We can also look for SSVEPs in the spectrogram, which uses color to represent the power of frequencies in the EEG signal over time

frequencies = np.logspace(1, 1.75, 60)
tfr, itc = tfr_morlet(epochs['30 Hz'], freqs=frequencies, picks='all',
                      n_cycles=15, return_itc=True)
tfr.plot(picks=[4], baseline=(-0.5, -0.1), mode='logratio',
         title='{0} - 30 Hz stim'.format(chs[4]));

tfr, itc = tfr_morlet(epochs['20 Hz'], freqs=frequencies, picks='all',
                      n_cycles=15, return_itc=True)
tfr.plot(picks=[4], baseline=(-0.5, -0.1), mode='logratio',
         title='{0} - 30 Hz stim'.format(chs[4]));

plt.tight_layout()

# Once again we can see clear SSVEPs at 30hz and 20hz
