# options
from postprocess.P300PostProcess import post_process_p300

is_plotting = True
is_exporting_samples = False

# user parameters ######################################################

tmin = -0.2
tmax = 0.5
resample_f = 50

# notes for plotting and explorting data
# notes = 'Galea unit 35 6/22'
# notes = 'Galea unit 37 6/24'
# notes = 'Galea unit 35 6/25'
# notes = 'Galea unit 39 6/25'
notes = 'P300_062221-062521'

participant = (0, 5)
data_path_list = [
    # 'C:/Dropbox/OpenBCI/Data/06-22-2021 7-09 PM Data.csv',  # Galea p#0
    # 'C:/Dropbox/OpenBCI/Data/06-22-2021 7-12 PM Data.csv',  # Galea p#0
    #
    # 'C:/Dropbox/OpenBCI/Data/06-24-2021 7-45 PM Data.csv',  # Galea p#5
    # 'C:/Dropbox/OpenBCI/Data/06-24-2021 7-43 PM Data.csv',  # Galea p#5
    #
    # 'C:/Dropbox/OpenBCI/Data/06-25-2021 11-07 AM Data.csv',  # Galea p#5
    # 'C:/Dropbox/OpenBCI/Data/06-25-2021 11-09 AM Data.csv',  # Galea p#5

    # 'C:/Dropbox/OpenBCI/Data/06-25-2021 5-01 PM Data.csv',  # Galea p#1
    # 'C:/Dropbox/OpenBCI/Data/06-25-2021 5-03 PM Data.csv',  # Galea p#1

    # 'C:/Dropbox/OpenBCI/Data/06-29-2021 5-54 PM Data.csv',  # Saline Cap p#3
    # 'C:/Dropbox/OpenBCI/Data/06-29-2021 5-55 PM Data.csv',  # Saline Cap p#3

    # 'C:/Dropbox/OpenBCI/Data/07-09-2021 7-43 PM Data.csv',  # ElectroCap p#6
    # 'C:/Dropbox/OpenBCI/Data/07-09-2021 8-16 PM Data.csv',  # ElectroCap p#6
    # 'C:/Dropbox/OpenBCI/Data/07-09-2021 8-18 PM Data.csv',  # ElectroCap p#6

    'C:/Users/S-Vec/Downloads/Galea MVT Data/10-15-2021 2-09 PM Data.csv'
]
headset = 'Galea'

post_process_p300(data_path_list, headset, participant, notes, is_plotting, is_exporting_samples, tmin, tmax, resample_f)