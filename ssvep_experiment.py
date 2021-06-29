import os
from eegnb import generate_save_fn
from eegnb.devices.eeg import EEG
from eegnb.experiments.visual_ssvep import ssvep

# Define some variables
board_name = "cyton_daisy"
experiment = "visual_ssvep"
subject_id = 3
session_nb = 0
record_duration = 120
serial_port='COM23'

eeg_device = EEG(device=board_name, serial_port=serial_port)
save_fn = generate_save_fn(board_name, experiment, subject_id, session_nb)
print(save_fn)
ssvep.present(duration=record_duration, eeg=eeg_device, save_fn=save_fn)
