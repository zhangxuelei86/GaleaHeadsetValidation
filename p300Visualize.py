from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne import Epochs, find_events
from mne.channels import make_standard_montage


# user parameters ######################################################

tmin = -0.2
tmax = 0.5
resample_f = 50

notes = 'Galea unit 35 6/22'
num_participant = 1
data_path_list = ['C:/Dropbox/OpenBCI/Data/06-22-2021 7-09 PM Data.csv',
                  'C:/Dropbox/OpenBCI/Data/06-22-2021 7-12 PM Data.csv']
headset = 'Galea'
event_id = {'Target': 2, 'Distractor': 1}
color_dict = {'Target': 'red', 'Distractor': 'blue'}

if headset == 'Galea':
    eeg_slice = np.s_[:, 7:17]  # first dimension is time, taking 6 eeg channels
    stim_slice = np.s_[:, -2]
    sampling_freq = 250  # in Hertz
    ch_names = ['Fp1', 'Fp2', 'Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'O1', 'O2']
    desired_chs = ['Pz', 'Cz', 'Fz']
else:
    eeg_slice = np.s_[:, 1:6]  # first dimension is time, taking 6 eeg channels
    stim_slice = np.s_[:, -2]
    sampling_freq = 250  # in Hertz
    ch_names = ['O1', 'O2', 'P3', 'Pz', 'P4']  # electrode cap does not have oz
    desired_chs = ['O1', 'O2', 'P3', 'Pz', 'P4']

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
raw = raw.filter(l_freq=0.5, h_freq=50, method='iir')
raw, _ = mne.set_eeg_reference(raw, 'average',projection=False)
raw.plot_psd(fmin=1, fmax=30)

events = find_events(raw)

# print('sample drop %: ', (1 - len(epochs.events) / len(events)) * 100)

for d_ch in desired_chs:
    matplotlib.pyplot.figure(figsize=(8.6, 7.2))
    for event_name, event_marker_id in event_id.items():
        time_vector = np.linspace(tmin, tmax, int((tmax - tmin) * resample_f))
        epochs = Epochs(raw, events=events, event_id=event_id,tmin=tmin, tmax=tmax, baseline=(-0.1, 0.0), preload=True,verbose=False, picks=d_ch,)
        y = -epochs.pick(d_ch)[event_name].resample(50).get_data().reshape((-1, len(time_vector)))  # flip the axis
        y1 = np.mean(y, axis=0) + 0.1 *np.std(y, axis=0)
        y2 = np.mean(y, axis=0) - 0.1 * np.std(y, axis=0)
        plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor=color_dict[event_name], interpolate=True, alpha=0.5)
        plt.plot(time_vector, np.mean(y, axis=0), c = color_dict[event_name], label=event_name)

    plt.xlabel('Time (sec)')
    plt.ylabel('Amplitude (\u03BCV)')
    plt.title('Ch {0}, {1}, #trial={2}, #sbj={3}'.format(d_ch, notes, events.shape[0], num_participant))
    plt.legend()
    plt.show()


