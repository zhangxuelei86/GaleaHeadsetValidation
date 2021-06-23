import os
from eegnb import generate_save_fn
from eegnb.devices.eeg import EEG
from eegnb.experiments.visual_p300 import p300

# Define some variables
board_name = "cyton"
experiment = "p300"
subject_id = 0
session_nb = 0
record_duration = 120

eeg_device = EEG(device=board_name, serial_port='COM23')

# Create save file name
save_fn = generate_save_fn(board_name, experiment, subject_id, session_nb)
print(save_fn)

p300.present(duration=record_duration, eeg=eeg_device, save_fn=save_fn)
