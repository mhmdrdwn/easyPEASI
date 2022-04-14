from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net


def build_model(config):
    if config.model_name == 'shallow':
        model = ShallowFBCSPNet(in_chans=config.n_chans, n_classes=config.n_classes,
                                n_filters_time=config.n_start_chans,
                                n_filters_spat=config.n_start_chans,
                                input_time_length=config.input_time_length,
                                final_conv_length=config.final_conv_length).create_network()
        
        model = ShallowFBCSPNet(in_chans=config.n_chans, n_classes=config.n_classes,
                        input_time_length=config.input_time_length,
                        final_conv_length=config.final_conv_length).create_network()
        
    elif config.model_name == 'deep':
        model = Deep4Net(config.n_chans, config.n_classes,
                         n_filters_time=config.n_start_chans,
                         n_filters_spat=config.n_start_chans,
                         input_time_length=config.input_time_length,
                         n_filters_2 = int(config.n_start_chans * config.n_chan_factor),
                         n_filters_3 = int(config.n_start_chans * (config.n_chan_factor ** 2.0)),
                         n_filters_4 = int(config.n_start_chans * (config.n_chan_factor ** 3.0)),
                         final_conv_length=config.final_conv_length,
                         stride_before_pool=True).create_network()
        
    elif (config.model_name == 'deep_smac'):
        if config.model_name == 'deep_smac':
            do_batch_norm = False
        else:
            assert config.model_name == 'deep_smac_bnorm'
            do_batch_norm = True
        double_time_convs = False
        drop_prob = 0.244445
        filter_length_2 = 12
        filter_length_3 = 14
        filter_length_4 = 12
        filter_time_length = 21
        final_conv_length = 1
        first_nonlin = elu
        first_pool_mode = 'mean'
        first_pool_nonlin = identity
        later_nonlin = elu
        later_pool_mode = 'mean'
        later_pool_nonlin = identity
        n_filters_factor = 1.679066
        n_filters_start = 32
        pool_time_length = 1
        pool_time_stride = 2
        split_first_layer = True
        n_chan_factor = n_filters_factor
        n_start_chans = n_filters_start
        model = Deep4Net(config.n_chans, config.n_classes,
                 n_filters_time=n_start_chans,
                 n_filters_spat=n_start_chans,
                 input_time_length=config.input_time_length,
                 n_filters_2=int(n_start_chans * n_chan_factor),
                 n_filters_3=int(n_start_chans * (n_chan_factor ** 2.0)),
                 n_filters_4=int(n_start_chans * (n_chan_factor ** 3.0)),
                 final_conv_length=config.final_conv_length,
                 batch_norm=do_batch_norm,
                 double_time_convs=double_time_convs,
                 drop_prob=drop_prob,
                 filter_length_2=filter_length_2,
                 filter_length_3=filter_length_3,
                 filter_length_4=filter_length_4,
                 filter_time_length=filter_time_length,
                 first_nonlin=first_nonlin,
                 first_pool_mode=first_pool_mode,
                 first_pool_nonlin=first_pool_nonlin,
                 later_nonlin=later_nonlin,
                 later_pool_mode=later_pool_mode,
                 later_pool_nonlin=later_pool_nonlin,
                 pool_time_length=pool_time_length,
                 pool_time_stride=pool_time_stride,
                 split_first_layer=split_first_layer,
                 stride_before_pool=True).create_network()
        
    elif config.model_name == 'shallow_smac':
        conv_nonlin = identity
        do_batch_norm = True
        drop_prob = 0.328794
        filter_time_length = 56
        final_conv_length = 22
        n_filters_spat = 73
        n_filters_time = 24
        pool_mode = 'max'
        pool_nonlin = identity
        pool_time_length = 84
        pool_time_stride = 3
        split_first_layer = True
        model = ShallowFBCSPNet(in_chans=config.n_chans, n_classes=config.n_classes,
                                n_filters_time=n_filters_time,
                                n_filters_spat=n_filters_spat,
                                input_time_length=input_time_length,
                                final_conv_length=config.final_conv_length,
                                conv_nonlin=conv_nonlin,
                                batch_norm=do_batch_norm,
                                drop_prob=drop_prob,
                                filter_time_length=filter_time_length,
                                pool_mode=pool_mode,
                                pool_nonlin=pool_nonlin,
                                pool_time_length=pool_time_length,
                                pool_time_stride=pool_time_stride,
                                split_first_layer=split_first_layer,
                                ).create_network()
    elif config.model_name == 'linear':
        model = nn.Sequential()
        model.add_module("conv_classifier",
                         nn.Conv2d(config.n_chans, config.n_classes, (600,1)))
        model.add_module('Softmax', nn.Softmax(1))
        model.add_module('squeeze', Expression(lambda x: x.squeeze(3)))
        
    #to_dense_prediction_model(model)
    if config.cuda:
        model.cuda()
    return model  
