# options
import pandas as pd
import os

from postprocess.P300PostProcess import post_process_p300

# user parameters ######################################################
from postprocess.RestStatePostProcess import post_process_rest_state
from postprocess.SSVEPPostProcess import post_process_ssvep

notes = 'Partial Test 11/9/20'
path_to_notes = 'C:/Users/S-Vec/Downloads/Galea MVT Data (1)/Galea MVT Data/notes.csv'
data_root = 'C:/Users/S-Vec/Downloads/Galea MVT Data (1)/Galea MVT Data'

is_plotting = True
is_exporting_samples = False

# experiment configurations ######################################################
p300_tmin = -0.2
p300_tmax = 0.5
ssvep_tmin = -0.5
ssvep_tmax = 4.0

tmin_focus = 0.2
tmax_focus = 9.8
tmin_rest = 0.2
tmax_rest = 29.8

timelocking_dict = {
    'Focus': [1, 9.8],
    'Rest': [1, 29.8]
}

resample_f = 50
df = pd.read_csv(path_to_notes)
# custom code after this
participants = pd.unique(df['Participant'])  # find the MVT

# for mvt in mvts:
    # process p300
    # P300_rows = df.loc[(df['MVT'] == mvt) & (df['Experiment'] == "P300")]
    # p300_data_path_list = [os.path.join(data_root, x) for x in P300_rows['FileName'].values]
    # post_process_p300(p300_data_path_list, 'Galea', 'x', notes, is_plotting, is_exporting_samples, p300_tmin, p300_tmax, resample_f)
    #
    # # process SSVEP
    # SSVEP_rows = df.loc[(df['MVT'] == mvt) & (df['Experiment'] == "SSVEP")]
    # SSVEP_data_path_list = [os.path.join(data_root, x) for x in SSVEP_rows['FileName'].values]
    # post_process_ssvep(SSVEP_data_path_list, 'Galea', 'x', notes, ssvep_tmin, ssvep_tmax)

    # process RestState
    # RestState_rows = df.loc[(df['Participant'] == '1') & (df['Experiment'] == "RestState")]
    # RestState_data_path_list = [os.path.join(data_root, x) for x in RestState_rows['FileName'].values]
    # post_process_rest_state({'Galea': RestState_data_path_list}, timelocking_dict, 'x', notes)

# SSVEP_rows = df.loc[(df['Experiment'] == "SSVEP")]
# SSVEP_data_path_list = [os.path.join(data_root, x) for x in SSVEP_rows['FileName'].values]
# post_process_ssvep(SSVEP_data_path_list, 'Galea', 'x', notes, ssvep_tmin, ssvep_tmax)

# process p300
P300_rows = df.loc[(df['Experiment'] == "P300")]
p300_data_path_list = [os.path.join(data_root, x) for x in P300_rows['FileName'].values]
post_process_p300(p300_data_path_list, 'Galea', 'x', notes, is_plotting, is_exporting_samples, p300_tmin, p300_tmax, resample_f)

# process RestState
# RestState_rows = df.loc[(df['Experiment'] == "RestState")]
# RestState_data_path_list = [os.path.join(data_root, x) for x in RestState_rows['FileName'].values]
# post_process_rest_state({'Galea': RestState_data_path_list}, timelocking_dict, 'x', notes)
