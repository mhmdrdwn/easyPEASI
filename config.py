class Config():
    data_folders = [
        './Data/normal/',
        './Data/abnormal/']
    n_recordings = 600  # number of edf files to analyse, if you want to restrict the set size
    sensor_types = ["EEG"]
    n_chans = 21 #number of channels
    max_recording_mins = 35  # exclude larger recordings from training set
    sec_to_cut = 60  # cut away at start of each recording
    duration_recording_mins = 5#20  # how many minutes to use per recording
    test_recording_mins = 5#20
    max_abs_val = 800  # for clipping
    sampling_freq = 100
    divisor = 10  # divide signal by this
    test_on_eval = True  # teston evaluation set or on training set
    n_folds = 10 #number of KFolds
    i_test_fold = 9
    shuffle = True
    model_name = 'deep'
    n_start_chans = 25
    n_chan_factor = 2  # relevant for deep model only
    input_time_length = 30000
    final_conv_length = 'auto'
    model_constraint = 'defaultnorm'
    init_lr = 1e-4
    batch_size = 16
    max_epochs = 11 # until first stop, the continue train on train+valid
    cuda = False
    n_classes = 2
