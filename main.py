import pandas as pd
import numpy as np
import mne
from mne.channels import make_standard_montage
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('Qt5Agg')

# plt.ion()
# user parameters
data_path = '/Users/Leo/Dropbox/OpenBCI/Data/06-04-2021 6-34 PM Data.csv'

eeg_slice = np.s_[:, 1:6]  # first dimension is time, taking 6 eeg channels
stim_slice = np.s_[:, -2:-1]
sampling_freq = 250  # in Hertz
# ch_names = ['O1', 'Oz', 'O3', 'P3', 'Pz', 'P4']
ch_names = ['O1', 'O2', 'P3', 'Pz', 'P4', 'stim']  # electrode cap does not have oz
eeg_n_ch = 5

data_df = pd.read_csv(data_path, header=None)

# grab the eeg data channels
eeg_array = data_df.iloc[eeg_slice].values
eeg_array = np.nan_to_num(eeg_array, 0)
stim_array = data_df.iloc[stim_slice].values
eeg_stim_array = np.concatenate([eeg_array, stim_array], axis=-1)

montage = make_standard_montage('standard_1005')
n_channels = len(ch_names) # plus one stim channel
ch_types = ['eeg'] * eeg_n_ch + ['stim']
info = mne.create_info(ch_names, sfreq=sampling_freq, ch_types=ch_types)
raw = mne.io.RawArray(eeg_stim_array.T, info)
raw.set_montage(montage)
# raw, _ = mne.set_eeg_reference(raw, 'average', projection=False)
raw = raw.filter(l_freq=2, h_freq=50)

events = mne.find_events(raw, ['stim'])

# note = 'No reference, no baseline '
note = 'Average reference, baseline -1, 0, Dist 1 '

f_freq = 30
title = note + 'Flickering Frequency: {0} Hz'.format(f_freq)
epochs_params = dict(events=events, event_id=[f_freq], tmin=-1.5, tmax=10, baseline=(-1.5, -0.5))
evoked = mne.Epochs(raw=raw, **epochs_params, preload=True)
# evoked.filter(l_freq=f_freq-5, h_freq=f_freq+5)
evoked.plot_psd(fmin=5, fmax=40, picks='eeg', tmin=0, tmax=10, show=False)
plt.title(title)
plt.show()
a = evoked.get_data()[0, :5, :]
f, t, Sxx = signal.spectrogram(a[ch_names.index('O1')], sampling_freq)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title(title)
plt.show()

f_freq = 20
title = note + 'Flickering Frequency: {0} Hz'.format(f_freq)
epochs_params = dict(events=events, event_id=[f_freq], tmin=-1.5, tmax=10, baseline=(-1.5, -0.5))
evoked = mne.Epochs(raw=raw, **epochs_params, preload=True)
# evoked.filter(l_freq=f_freq-5, h_freq=f_freq+5)
evoked.plot_psd(fmin=5, fmax=40, picks='eeg', tmin=0, tmax=10, show=False)
plt.title(title)
plt.show()
a = evoked.get_data()[0, :5, :]
f, t, Sxx = signal.spectrogram(a[ch_names.index('O1')], sampling_freq)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title(title)
plt.show()

# a = evoked.get_data()
# a = a[0]
# [plt.plot(a[i, :]) for i in range(len(a)-1)]
# plt.plot(a[0, :][:500])
# plt.show()

# f_freq = 20
# title = note + 'Flickering Frequency: {0} Hz'.format(f_freq)
# epochs_params = dict(events=events, event_id=[f_freq], tmin=-1.5, tmax=10, baseline=(-1, 0))
# evoked = mne.Epochs(raw=raw, **epochs_params)
# evoked.plot_psd(fmin=5, fmax=40, picks='eeg', tmin=0, tmax=10, show=False)
# plt.title(title)
# plt.show()
# a = evoked.get_data()[0, :5, :]
# f, t, Sxx = signal.spectrogram(a[ch_names.index('O1')], sampling_freq)
# plt.pcolormesh(t, f, Sxx, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.title(title)
# plt.show()
#
# f_freq = 30
# title = note + 'Flickering Frequency: {0} Hz'.format(f_freq)
# epochs_params = dict(events=events, event_id=[f_freq], tmin=-1.5, tmax=10, baseline=(-1, 0))
# evoked = mne.Epochs(raw=raw, **epochs_params)
# evoked.plot_psd(fmin=5, fmax=40, picks='eeg', tmin=0, tmax=10, show=False)
# plt.title(title)
# plt.show()
# a = evoked.get_data()[0, :5, :]
# f, t, Sxx = signal.spectrogram(a[ch_names.index('O1')], sampling_freq)
# plt.pcolormesh(t, f, Sxx, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.title(title)
# plt.show()