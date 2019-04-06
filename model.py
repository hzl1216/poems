#-*- coding: UTF-8 -*-
import torch,math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        #The contiguous() function turns tensor into a continuous distribution in memory.
        return x[:, :, :-self.chomp_size].contiguous()

class AttentionBlock(nn.Module):
  """An attention mechanism similar to Vaswani et al (2017)
  The input of the AttentionBlock is `BxTxD` where `B` is the input
  minibatch size, `T` is the length of the sequence `D` is the dimensions of
  each feature.
  The output of the AttentionBlock is `BxTx(D+V)` where `V` is the size of the
  attention values.
  Arguments:
      dims (int): the number of dimensions (or channels) of each element in
          the input sequence
      k_size (int): the size of the attention keys
      v_size (int): the size of the attention values
      seq_len (int): the length of the input and output sequences
  """
  def __init__(self, dims, k_size, v_size, seq_len=None):
    super(AttentionBlock, self).__init__()
    self.key_layer = nn.Linear(dims, k_size)
    self.query_layer = nn.Linear(dims, k_size)
    self.value_layer = nn.Linear(dims, v_size)
    self.sqrt_k = math.sqrt(k_size)

  def forward(self, minibatch):
    minibatch = minibatch.transpose(2,1)
    keys = self.key_layer(minibatch)
    queries = self.query_layer(minibatch)
    values = self.value_layer(minibatch)
    logits = torch.bmm(queries, keys.transpose(2,1))
    # Use numpy triu because you can't do 3D triu with PyTorch
    # TODO: using float32 here might break for non FloatTensor inputs.
    # Should update this later to use numpy/PyTorch types of the input.
    mask = np.triu(np.ones(logits.size()), k=1).astype('uint8')
    mask = torch.from_numpy(mask).cuda()
    # do masked_fill_ on data rather than Variable because PyTorch doesn't
    # support masked_fill_ w/-inf directly on Variables for some reason.
    logits.data.masked_fill_(mask, float('-inf'))
    probs = F.softmax(logits, dim=1) / self.sqrt_k
    read = torch.bmm(probs, values)
    return minibatch + read

#Residual convolution
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # move padding
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

#define Multi-layer residual network
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2,d_size=2,attention=False,max_length=256):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = d_size ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        if attention == True:
            print('add attention layer')
            layers += [AttentionBlock(max_length, 256, 256)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):

    def __init__(self, input_size, output_size, num_channels,attention=False,
                 kernel_size=2, dropout=0.2, emb_dropout=0.1, tied_weights=False):
        super(TCN, self).__init__()
        self.attention = attention
        # word embedding
        self.encoder = nn.Embedding(output_size, input_size)
        # input to Multi-layer residual network
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout,attention=attention)
        # deconder to word
        self.decoder = nn.Linear(num_channels[-1], output_size)
        if tied_weights:
            if num_channels[-1] != input_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        emb = self.drop(self.encoder(input))
        y = self.tcn(emb.transpose(1, 2))
        if self.attention == False:
            y = self.tcn(emb.transpose(1, 2)).transpose(2,1)
        y = self.decoder(y)
        return y.contiguous()
