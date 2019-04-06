#-*- coding: UTF-8 -*-
import argparse
import matplotlib.pyplot as plt
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
sys.path.append("../../")
from model import *
from tqdm import tqdm
from pre_data import *
import pickle
from random import randint


parser = argparse.ArgumentParser(description='Jack Hong create poetry ！！！！')

parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size (default: 64)')
parser.add_argument('--cuda', type=bool, default=False,
                    help='use CUDA (default: False)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=2,
                    help='kernel size (default: 2)')
parser.add_argument('--emsize', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--test', type=bool, default=False,
                    help='train or test (default: False)')
parser.add_argument('--n', type=int, default=1,
                    help='generate the count of poems (default: 1)')
parser.add_argument('--startwords', type=str, nargs="+", default=None,
                    help='Tibetan poetry (default: None)')
parser.add_argument('--attention', type=bool, default=False,
                    help='is or not add attention (default: False)')

args = parser.parse_args()

#set the channel of Multi-layer residual network
num_channels = [args.emsize]*args.levels
n_words = len(word_num_map)

num_epochs = args.epochs
batch_size = args.batch_size
n_chunk = len(poetrys_vector) // batch_size
dataSet = DataSet(len(poetrys_vector))
net = TCN(args.emsize,n_words,num_channels,attention=args.attention)
if args.cuda:
    net = nn.DataParallel(net)
    net.cuda()
    print('using gpu #', torch.cuda.current_device())


def train():
    try:
        with open("model/model.pt", 'rb') as f:
            global net
            net = torch.load(f)
            print('load net,continue train')
    except:
        pass
    running_loss = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)
    for epoch in tqdm(range(num_epochs)):
        running_loss_tmp =0
        for step in range(n_chunk):
            x, y = dataSet.next_batch(batch_size)
            if args.cuda:
                x = Variable(torch.from_numpy(x)).long().cuda()
                y = Variable(torch.from_numpy(y)).long().cuda()
            else:
                x = Variable(torch.from_numpy(x)).long()
                y= Variable(torch.from_numpy(y)).long()
            output= net(x)
            output = output.contiguous().view(-1, n_words)

            y = y.contiguous().view(-1)
            _, idx = torch.max(output.detach(), 1)
            # print(y)
            optimizer.zero_grad()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            running_loss_tmp += loss.data
        with open("model/model.pt", 'wb') as f:
            print('Save model!\n')
            torch.save(net, f)
        print('epoch', epoch, ':loss is', running_loss_tmp/n_chunk)
        running_loss.append(running_loss_tmp.cpu().numpy()/n_chunk)
    print('training finished', ':loss is', np.mean(running_loss))
    x1 = range(0,num_epochs)
    y1 = running_loss
    plt.plot(x1, y1, 'o-')
    plt.xlim((0,num_epochs))
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.savefig("result/test_loss.jpg")
    plt.close()

def to_word(output,max=1):
    output = output[:, -1, :]
    output = torch.reshape(output, [-1])
    output = torch.nn.functional.softmax(output,dim=0)
    # get n words with the highest probability
    _, idx = torch.topk(output.detach(), max)
    index = idx[(int(torch.rand(1) * max))]
    if idx[0] in range(0,4):
        index = idx[0]
    return words[index]

def weigh(poems,times=10):
    print(poems)
    result = 0
    x = [2]*(len(poems)+1)
    x[1:] = [word_num_map[word] for word in poems]
    input = np.reshape(x, [1, -1])
    if args.cuda:
        input = Variable(torch.from_numpy(input)).long().cuda()
    else:
        input = Variable(torch.from_numpy(input)).long()
    for i in range(times):
        output = net(input)
        output = torch.nn.functional.softmax(output,dim=2)
        output = torch.reshape(output, [*output.shape[1:]])
        if type(result) == int:
            result = output
        else:
            result +=  (output-result)/(i+1)
    _, idx = torch.topk(result.detach(), k=10,dim=1)
    newpoems =''
    for id in range(len(idx)-1):
        if word_num_map[poems[id]] in idx[id]:
            newpoems+=poems[id]
        else:
            newpoems+=words[idx[id][(int(torch.rand(1) * 10))]]
    return newpoems
def test(n,startword = None):
    with open("model/model.pt", 'rb') as f:
        global net
        net = torch.load(f)
        print('load net')

    def poem(startword = None):
        step = 1
        if startword:
            word = startword[0]
        else:
            # Randomly obtain one of the 100 words with the highest probability
            x = np.array([list(map(word_num_map.get, '['))])
            if args.cuda:
                x = Variable(torch.from_numpy(x)).long().cuda()
            else:
                x = Variable(torch.from_numpy(x)).long()
            output = net(x)
            word = to_word(output, 100)
        x = [2]
        poem= ''
        # ']'  is the terminator sign
        while word != ']' :
            if startword:
                # set the first word of each sentence
                if word == '。'  :
                    if step < len(startword):
                        poem += word
                        x = [2]
                        word = startword[step]
                        step+=1
                    else:
                        poem += word
                        break
            poem += word
            x.append(word_num_map[poem[-1]])
            input = np.reshape(x, [1, -1])
            if args.cuda:
                input = Variable(torch.from_numpy(input)).long().cuda()
            else:
                input = Variable(torch.from_numpy(input)).long()
            output = net(input)
            # Randomly obtain one of the n words with the highest probability
            word = to_word(output, 3)
        return poem
    poems = []
    for i in range(n):
        if startword:
            poems.append(poem(startword))
        else:
            poems.append(weigh(poem(),10))
    return  poems


if __name__ == "__main__":
    if args.test:
        if args.startwords:
 #           Tibetan poetry 藏头诗 such as ****** python main.py --test=True --startwords 一 二 三
            for poem in test(args.n, startword=args.startwords):
                if len(set (len(line) for line in poem.split('。'))) ==2:
                    print(poem)
        else:
            for poem in test(args.n):
                print(poem)

    else:
        train()


