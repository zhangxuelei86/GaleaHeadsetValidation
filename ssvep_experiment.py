import os
from eegnb import generate_save_fn
from eegnb.devices.eeg import EEG
from eegnb.experiments.visual_ssvep import ssvep

# Define some variables
board_name = "cyton"
experiment = "visual_ssvep"
subject_id = 0
session_nb = 0
record_duration = 120
serial_port='COM7'

eeg_device = EEG(device=board_name)
save_fn = generate_save_fn(board_name, experiment, subject_id, session_nb)
print(save_fn)