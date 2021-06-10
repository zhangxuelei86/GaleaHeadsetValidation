import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne import Epochs
from mne.channels import make_standard_montage
from mne.time_frequency import psd_welch

# EEG-Notebooks functions

# user parameters ######################################################
data_path = 'C:/Dropbox/OpenBCI/Data/06-05-2021 5-13 PM Data.csv'
headset = 'Galea'
# end of user parameters ######################################################

if headset == 'Galea':
    eeg_slice = np.s_[:, 7:17]  # first dimension is time, taking 6 eeg channels
    stim_slice = np.s_[:, -2]
    sampling_freq = 250  # in Hertz
    ch_names = ['Fp1', 'Fp2', 'Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'O1', 'O2']
    desired_chs = ['O1', 'Oz', 'O2', 'P3', 'Pz', 'P4']
else:
    eeg_slice = np.s_[:, 1:6]  # first dimension is time, taking 6 eeg channels
    stim_slice = np.s_[:, -2]
    sampling_freq = 250  # in Hertz
    ch_names = ['O1', 'O2', 'P3', 'Pz', 'P4']  # electrode cap does not have oz
    desired_chs = ['O1', 'O2', 'P3', 'Pz', 'P4']

data_df = pd.read_csv(data_path)

# grab the eeg data channels
eeg_array = data_df.iloc[eeg_slice].values
eeg_array = np.nan_to_num(eeg_array, nan=0)
stim_array = np.expand_dims(data_df.iloc[stim_slice].values, axis=-1)
stim_array = np.nan_to_num(stim_array, nan=0)
eeg_stim_array = np.concatenate([eeg_array, stim_array], axis=-1)

montage = make_standard_montage('standard_1005')
eeg_n_ch = len(ch_names)
ch_types = ['eeg'] * eeg_n_ch + ['stim']
ch_names += ['EventMarker']
info = mne.create_info(ch_names, sfreq=sampling_freq, ch_types=ch_types)
raw = mne.io.RawArray(eeg_stim_array.T, info)
raw.set_montage(montage)
# raw = raw.filter(l_freq=2, h_freq=50)

# plot power spectrum density
raw.plot_psd(picks=desired_chs, show=False)  # for Galea
plt.title('Headset {0}: Power Density Spectrum for {1}'.format(headset, data_path))
plt.show()

# plot events
events = mne.find_events(raw)
event_id = {'30 Hz': 30, '20 Hz': 20}
epochs = Epochs(raw, events=events, event_id=event_id,
                tmin=-0.5, tmax=10, baseline=None, preload=True,
                verbose=False, picks='eeg')
print('sample drop %: ', (1 - len(epochs.events) / len(events)) * 100)


f, axs = plt.subplots(2, 1, figsize=(10, 10))
psd1, freq1 = psd_welch(epochs['30 Hz'], n_fft=1028, n_per_seg=256 * 3, picks=desired_chs)
psd2, freq2 = psd_welch(epochs['20 Hz'], n_fft=1028, n_per_seg=256 * 3, picks=desired_chs)
psd1 = 10 * np.log10(psd1)
psd2 = 10 * np.log10(psd2)

psd1_mean = psd1.mean(0)
psd1_std = psd1.mean(0)

psd2_mean = psd2.mean(0)
psd2_std = psd2.mean(0)

axs[0].plot(freq1, psd1_mean[[1, 4], :].mean(0), color='b', label='30 Hz')
axs[0].plot(freq2, psd2_mean[[1, 4], :].mean(0), color='r', label='20 Hz')

axs[1].plot(freq1, psd1_mean[4, :], color='b', label='30 Hz')
axs[1].plot(freq2, psd2_mean[4, :], color='r', label='20 Hz')

axs[0].set_title('Headset {0}: Oz and Pz'.format(headset))
axs[1].set_title('Headset {0}: P4'.format(headset))

axs[0].set_ylabel('Power Spectral Density (dB)')
axs[1].set_ylabel('Power Spectral Density (dB)')

axs[0].set_xlim((2, 50))
axs[1].set_xlim((2, 50))

axs[1].set_xlabel('Frequency (Hz)')

axs[0].legend()
axs[1].legend()

plt.show()

# f_freq = 30
# title = note + 'Flickering Frequency: {0} Hz'.format(f_freq)
# epochs_params = dict(events=events, event_id=[f_freq], tmin=-1.5, tmax=10, baseline=(-1.5, -0.5))
# evoked = mne.Epochs(raw=raw, **epochs_params, preload=True)
# # evoked.filter(l_freq=f_freq-5, h_freq=f_freq+5)
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
# f_freq = 20
# title = note + 'Flickering Frequency: {0} Hz'.format(f_freq)
# epochs_params = dict(events=events, event_id=[f_freq], tmin=-1.5, tmax=10, baseline=(-1.5, -0.5))
# evoked = mne.Epochs(raw=raw, **epochs_params, preload=True)
# # evoked.filter(l_freq=f_freq-5, h_freq=f_freq+5)
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
