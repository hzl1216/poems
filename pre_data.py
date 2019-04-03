#-*- coding: UTF-8 -*-
import collections
import numpy as np
import  os,json
from torch.utils.data import Dataset,DataLoader
'''
train_poetry_model.py 生成古诗模型 win10 python3.6.1 tensorflow1.2.1
'''
#-------------------------------数据预处理---------------------------#

poetry_file ='data/poetry.txt'
word_num_map_path = 'data/map.txt'

# 诗集
poetrys = []
with open(poetry_file, "r", encoding = 'utf-8') as f:
    for line in f:
        try:
            #line = line.decode('UTF-8')
            line = line.strip(u'\n')
            title, content = line.strip(u' ').split(u':')
            content = content.replace(u' ',u'')
            if u'_' in content or u'(' in content or u'（' in content or u'《' in content or u'[' in content:
                continue
            if len(content) < 5 or len(content) > 79:
                continue
            content = u'[' + content + u']'
            poetrys.append(content)
        except Exception as e:
            pass

# 按诗的字数排序
if os.path.exists(word_num_map_path) == False:
    poetrys = sorted(poetrys, key = lambda line: len(line))

    print('唐诗总数: ', len(poetrys))
    # 统计每个字出现次数
    all_words = []
    for poetry in poetrys:
        all_words += [word for word in poetry]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)

    # 取前多少个常用字
    words = words[:len(words)] + (' ',)
    # 每个字映射为一个数字ID
    word_num_map = dict(zip(words, range(len(words))))
    # 把诗转换为向量形式
    to_num = lambda word: word_num_map.get(word, len(words))
    poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys]
    with open(word_num_map_path, 'w') as f:
        jsObj = json.dumps(word_num_map)
        f.write(jsObj)
else:
    with open(word_num_map_path, 'r') as f:
        word_num_map = json.load(f)
        to_num = lambda word: word_num_map.get(word, len(word_num_map))
        poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]



batch_size = 64
n_chunk = len(poetrys_vector) // batch_size

class DataSet(object):
    def __init__(self, data_size):
        self._data_size = data_size
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._data_index = np.arange(data_size)

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        if start + batch_size > self._data_size:
            np.random.shuffle(self._data_index)
            self._epochs_completed = self._epochs_completed + 1
            self._index_in_epoch = batch_size
            full_batch_features, full_batch_labels = self.data_batch(0, batch_size)
            return full_batch_features, full_batch_labels
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            full_batch_features ,full_batch_labels = self.data_batch(start, end)
            if self._index_in_epoch == self._data_size:
                self._index_in_epoch = 0
                self._epochs_completed = self._epochs_completed + 1
                np.random.shuffle(self._data_index)
            return full_batch_features,full_batch_labels

    def data_batch(self,start,end):
        batches = []
        for i in range(start,end):
            batches.append(poetrys_vector[self._data_index[i]])

        length = max(map(len,batches))

        xdata = np.full((end - start, length), word_num_map[' '], np.int32)
        for row in range(end - start):
            xdata[row,:len(batches[row])] = batches[row]
        ydata = np.copy(xdata)
        ydata[:,:-1] = xdata[:, 1:]
        return xdata, ydata

