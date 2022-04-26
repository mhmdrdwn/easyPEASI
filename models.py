"""The models to be used for training"""

from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
import torch.nn as nn
import torch
from torch.autograd import Variable


"""Implimenation of the ChronoNet is https://github.com/talhaanwarch/youtube-tutorials"""
class Block(nn.Module):
  def __init__(self,inplace):
    super().__init__()
    self.conv1=nn.Conv1d(in_channels=inplace,out_channels=32,kernel_size=2,stride=2,padding=0)
    self.conv2=nn.Conv1d(in_channels=inplace,out_channels=32,kernel_size=4,stride=2,padding=1)
    self.conv3=nn.Conv1d(in_channels=inplace,out_channels=32,kernel_size=8,stride=2,padding=3)
    self.relu=nn.ReLU()

  def forward(self,x):
    x1=self.relu(self.conv1(x))
    x2=self.relu(self.conv2(x))
    x3=self.relu(self.conv3(x))
    x=torch.cat([x1,x3,x3],dim=1)
    return x

class ChronoNet(nn.Module):
  def __init__(self,channel):
    super().__init__()
    self.block1=Block(channel)
    self.block2=Block(96)
    self.block3=Block(96)
    self.gru1=nn.GRU(input_size=96,hidden_size=32,batch_first=True)
    self.gru2=nn.GRU(input_size=32,hidden_size=32,batch_first=True)
    self.gru3=nn.GRU(input_size=64,hidden_size=32,batch_first=True)
    self.gru4=nn.GRU(input_size=96,hidden_size=32,batch_first=True)
    self.gru_linear=nn.Linear(3750,1)
    self.flatten=nn.Flatten()
    self.fc1=nn.Linear(32,1)
    self.relu=nn.ReLU()

  def forward(self,x):
    x = x.squeeze()
    x=self.block1(x)
    x=self.block2(x)
    x=self.block3(x)
    x=x.permute(0,2,1)
    gru_out1,_=self.gru1(x)
    gru_out2,_=self.gru2(gru_out1)
    gru_out=torch.cat([gru_out1,gru_out2],dim=2)
    gru_out3,_=self.gru3(gru_out)
    gru_out=torch.cat([gru_out1,gru_out2,gru_out3],dim=2)
    #print('gru_out',gru_out.shape)
    linear_out=self.relu(self.gru_linear(gru_out.permute(0,2,1)))
    gru_out4,_=self.gru4(linear_out.permute(0,2,1))
    x=self.flatten(gru_out4)
    x=self.fc1(x)
    return x



def build_model(config, model_name):
    if model_name == 'shallow':
        model = ShallowFBCSPNet(in_chans=config.n_chans, 
                                n_classes=config.n_classes,
                                input_time_length=config.input_time_length,
                                final_conv_length=config.final_conv_length).create_network()
        
    elif model_name == 'deep':
        model = Deep4Net(in_chans=config.n_chans, n_classes=config.n_classes,
                         input_time_length=config.input_time_length,
                         n_filters_2 = int(config.n_start_chans * config.n_chan_factor),
                         n_filters_3 = int(config.n_start_chans * (config.n_chan_factor ** 2.0)),
                         n_filters_4 = int(config.n_start_chans * (config.n_chan_factor ** 3.0)),
                         final_conv_length=config.final_conv_length,
                         stride_before_pool=True).create_network()
        
    elif model_name == 'chronoNet':
        model=ChronoNet(config.n_chans)
    
    elif model_name == 'linear':
        model = nn.Sequential()
        model.add_module("conv_classifier",
                         nn.Conv2d(config.n_chans, config.n_classes, (600,1)))
        model.add_module('Softmax', nn.Softmax(1))
        model.add_module('squeeze', Expression(lambda x: x.squeeze(3)))
        
    #to_dense_prediction_model(model)
    if config.cuda:
        model.cuda()
    return model  
