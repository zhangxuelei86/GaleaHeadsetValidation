from postprocess.SSVEPPostProcess import post_process_ssvep

notes = 'SSVEP_061121-062521'

participants = (0,)

data_path_list = [
    # 'C:/Dropbox/OpenBCI/Data/06-11-2021 2-27 PM Data.csv',  # Galea p#?
    # 'C:/Dropbox/OpenBCI/Data/06-11-2021 2-44 PM Data.csv',  # Galea p#?

    # 'C:/Dropbox/OpenBCI/Data/06-11-2021 5-02 PM Data.csv',  # Galea p#0
    # 'C:/Dropbox/OpenBCI/Data/06-11-2021 5-13 PM Data.csv',  # Galea p#0
    #
    # 'C:/Dropbox/OpenBCI/Data/06-11-2021 5-41 PM Data.csv',  # Galea p#0
    # 'C:/Dropbox/OpenBCI/Data/06-11-2021 6-33 PM Data.csv',  # Galea p#0
    #
    # 'C:/Dropbox/OpenBCI/Data/06-11-2021 6-53 PM Data.csv',  # Galea p#0
    # 'C:/Dropbox/OpenBCI/Data/06-11-2021 7-04 PM Data.csv',  # Galea p#0

    # 'C:/Dropbox/OpenBCI/Data/06-25-2021 11-13 AM Data.csv',  # Galea p#5
    # 'C:/Dropbox/OpenBCI/Data/06-25-2021 12-43 PM Data.csv',  # Galea p#5

    # 'C:/Dropbox/OpenBCI/Data/06-25-2021 1-24 PM Data.csv',  # Galea p#5
    # 'C:/Dropbox/OpenBCI/Data/06-25-2021 4-35 PM Data.csv',  # Galea p#5
    'C:/Dropbox/OpenBCI/Data/06-29-2021 5-51 PM Data.csv',  # Saline Cap p#3
]
headset = 'Ultracortex'
tmin=0.2
tmax=3.8

post_process_ssvep(data_path_list, headset, participants, notes, tmin, tmax)