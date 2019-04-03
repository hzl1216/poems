import argparse
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

parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--cuda', type=bool, default=False,
                    help='use CUDA (default: False)')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit (default: 200)')
parser.add_argument('--ksize', type=int, default=2,
                    help='kernel size (default: 2)')
parser.add_argument('--emsize', type=int, default=256,
                    help='size of word embeddings')
parser.add_argument('--levels', type=int, default=6,
                    help='# of levels (default: 4)')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='initial learning rate (default: 4)')
parser.add_argument('--test', type=bool, default=False,
                    help='train or test (default: False)')
parser.add_argument('--n', type=int, default=1,
                    help='')
args = parser.parse_args()
num_channels = [args.emsize]*args.levels
n_words = len(word_num_map)
net = TCN(args.emsize,n_words,num_channels)
if args.cuda:
    net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)
trainds = DataSet(len(poetrys_vector))
running_loss = []
num_epochs = args.epochs

def train():
    try:
        with open("model/model.pt", 'rb') as f:
            global net
            net = torch.load(f)
            print('load net')
    except:
        pass

    for epoch in tqdm(range(num_epochs)):
        running_loss_tmp =0
        for step in range(n_chunk):
            x, y = trainds.next_batch(batch_size)
            x = Variable(torch.from_numpy(x)).long()
            y= Variable(torch.from_numpy(y)).long()
            output= net(x)
            output = output.contiguous().view(-1, n_words)

            y = y.contiguous().view(-1)
            _, idx = torch.max(output.detach(), 1)
            # print(y)
            optimizer.zero_grad()
            loss = criterion(output, y)
            print(loss)
            loss.backward()
            optimizer.step()
            running_loss_tmp += loss.data
        with open("model/model.pt", 'wb') as f:
            print('Save model!\n')
            torch.save(net, f)
        print('epoch', epoch, ':loss is', running_loss_tmp)
        running_loss.append(running_loss_tmp)
    print('training finished', ':loss is', np.mean(running_loss))

def to_word(output,max=1):
    output = output[:, -1, :]
    output = torch.nn.functional.softmax(output,dim=1)
    output = torch.reshape(output, [-1])

    _, idx = torch.topk(output.detach(), max)
    index = idx[(int(torch.rand(1) * max))]
    if idx[0] in range(0,4):
        index = idx[0]
    return words[index]
def test(n,startword = None):
    with open("model/model.pt", 'rb') as f:
        global net
        net = torch.load(f)
        print('load net')
    def poem(startword = None):
        if startword:
            word = startword
        else:
            x = np.array([list(map(word_num_map.get, '['))])
            x = Variable(torch.from_numpy(x)).long()
            output = net(x)
            word = to_word(output, 100)
        x = [2]
        poem= ''
        while word != ']' and len(poem)<50:
            poem += word
            x.append(word_num_map[poem[-1]])
            input = np.reshape(x, [1, -1])
            input = Variable(torch.from_numpy(input)).long()
            output = net(input)
            word = to_word(output)
            # word = words[np.argmax(probs_)]
        return poem
    def poem(startword = None):
        step = 1
        if startword:
            word = startword[0]
        else:
            x = np.array([list(map(word_num_map.get, '['))])
            x = Variable(torch.from_numpy(x)).long()
            output = net(x)
            word = to_word(output, 100)
        x = [2]
        poem= ''
        while word != ']' :
            if startword:
                if (word == '，'  or word == '。' ) and step < len(startword):
                    poem += word
                    word = startword[step]
                    step+=1
            poem += word
            x.append(word_num_map[poem[-1]])
            input = np.reshape(x, [1, -1])
            input = Variable(torch.from_numpy(input)).long()
            output = net(input)
            word = to_word(output, 10)
            # word = words[np.argmax(probs_)]
        return poem
    poems = []
    for i in range(n):
        if startword:
            poems.append(poem(startword))
        else:
            poems.append(poem())
    return  poems


if __name__ == "__main__":
    if args.test:
        # , startword = ['李', '子', '娇']
        for poem in test(10):
            print(poem)
    else:
        train()


