import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne import Epochs
from mne.channels import make_standard_montage
from mne.time_frequency import psd_welch

# EEG-Notebooks functions

# user parameters ######################################################
from utils import get_n_distinct_colors

notes = 'Galea unit 35 6/22'
num_participant = 1
data_path_list = ['C:/Dropbox/OpenBCI/Data/06-22-2021 12-31 PM Data.csv',
                   'C:/Dropbox/OpenBCI/Data/06-22-2021 1-01 PM Data.csv']
headset = 'Galea'
# event_id = {'30 Hz': 30, '20 Hz': 20}
event_id = {'15 Hz': 15, '12 Hz': 12}

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


# grab the eeg data channels
raw_list = []

for data_path in data_path_list:
    data_df = pd.read_csv(data_path)

    eeg_array = data_df.iloc[eeg_slice].values
    eeg_array = np.nan_to_num(eeg_array, nan=0)
    stim_array = np.expand_dims(data_df.iloc[stim_slice].values, axis=-1)
    stim_array = np.nan_to_num(stim_array, nan=0)
    eeg_stim_array = np.concatenate([eeg_array, stim_array], axis=-1)

    eeg_n_ch = len(ch_names)
    ch_types = ['eeg'] * eeg_n_ch + ['stim']
    info = mne.create_info(ch_names + ['EventMarker'], sfreq=sampling_freq, ch_types=ch_types)
    raw_list.append(mne.io.RawArray(eeg_stim_array.T, info))

raw = mne.concatenate_raws(raw_list)
montage = make_standard_montage('standard_1005')
raw.set_montage(montage)
# raw = raw.filter(l_freq=2, h_freq=50)

# plot power spectrum density
raw.plot_psd(picks=desired_chs, show=False)  # for Galea
plt.title('Headset {0}: Power Density Spectrum for {1}'.format(headset, data_path))
plt.show()

# plot events
events = mne.find_events(raw)
epochs = Epochs(raw, events=events, event_id=event_id,
                tmin=-0.5, tmax=4, baseline=None, preload=True,
                verbose=False, picks='eeg')
print('sample drop %: ', (1 - len(epochs.events) / len(events)) * 100)

for i, chan_name in enumerate(desired_chs):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_prop_cycle(color=get_n_distinct_colors(len(desired_chs)))
    for key, value in event_id.items():
        psd, freq = psd_welch(epochs[key], n_fft=1028, n_per_seg=256 * 3, picks=desired_chs)
        psd = 10 * np.log10(psd)
        psd_mean = psd.mean(0)
        psd_std = psd.mean(0)

        ax.plot(freq, psd_mean[[i, ], :].mean(0), label=key)
        ax.set_title('Headset {0}: {1}'.format(headset, chan_name))
        ax.set_ylabel('Power Spectral Density (dB)')
        ax.set_xlim((2, 50))
        ax.legend()
    ax.set_title('{0} {1} #trial={2}, #sub={3}'.format(chan_name, notes, len(epochs[list(event_id.keys())]), num_participant))
    plt.show()
