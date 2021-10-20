# options
import pandas as pd
import os

from postprocess.P300PostProcess import post_process_p300

# user parameters ######################################################
from postprocess.SSVEPPostProcess import post_process_ssvep

notes = 'MVT Spring Partial 10/15/20'
path_to_notes = 'C:/Dropbox/OpenBCI/Galea MVT Data/notes.csv'
data_root = 'C:/Dropbox/OpenBCI/Galea MVT Data'

is_plotting = True
is_exporting_samples = False

# experiment configurations ######################################################
p300_tmin = -0.2
p300_tmax = 0.5
ssvep_tmin = -0.5
ssvep_tmax = 4.0

resample_f = 50

df = pd.read_csv(path_to_notes)

mvts = pd.unique(df['MVT'])  # find the MVT



for mvt in mvts:
    # process p300
    P300_rows = df.loc[(df['MVT'] == mvt) & (df['Experiment'] == "P300")]
    p300_data_path_list = [os.path.join(data_root, x) for x in P300_rows['FileName'].values]
    post_process_p300(p300_data_path_list, 'Galea', 'x', notes, is_plotting, is_exporting_samples, p300_tmin, p300_tmax, resample_f)

    SSVEP_rows = df.loc[(df['MVT'] == mvt) & (df['Experiment'] == "SSVEP")]
    SSVEP_data_path_list = [os.path.join(data_root, x) for x in SSVEP_rows['FileName'].values]
    post_process_ssvep(SSVEP_data_path_list, 'Galea', 'x', notes, ssvep_tmin, ssvep_tmax)