from postprocess.RestStatePostProcess import post_process_rest_state
from postprocess.SSVEPPostProcess import post_process_ssvep

notes = 'RestState_070921-070921'

participants = (0,)

data_path_dict = {
    'Galea-ShortCurved': ['C:/Dropbox/OpenBCI/Data/07-09-2021 4-39 PM Data.csv',
                          'C:/Dropbox/OpenBCI/Data/07-09-2021 4-59 PM Data.csv',
                          ],
    # 'Ultracortex': [
    #     'C:/Dropbox/OpenBCI/Data/07-09-2021 5-14 PM Data.csv',
    #
    # ]

    # 'ElectroCap': [
    #     'C:/Dropbox/OpenBCI/Data/07-09-2021 7-37 PM Data.csv',
    #     'C:/Dropbox/OpenBCI/Data/07-09-2021 7-46 PM Data.csv',
    # ]
}
tmin_focus = 0.2
tmax_focus = 9.8
tmin_rest = 0.2
tmax_rest = 29.8

timelocking_dict = {
    'Focus': [1, 9.8],
    'Rest': [1, 29.8]
}

post_process_rest_state(data_path_dict, timelocking_dict, participants, notes)
