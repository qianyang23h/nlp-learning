import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

dtype = torch.FloatTensor

# data preprocessing

sentences = ['pig cat animal', 'cat pig animal', 'orange apple fruit'
             'Riemann like fruit', 'Riemann like orange and apple',
             'Riemann is a cat', 'sleep eat study action', 'Riemann like eating']

word_list = " ".join(sentences).split()
vocab = list(set(word_list))
word2idx = {w:i for i, w in enumerate(vocab)}
idx2word = {i:w for i, w in enumerate(vocab)}


# parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 8
C = 2  # windows size
vocab_size = len(vocab)
embed_size = 8


skip_gram = []  # [[center1, context1], [center1, context2], [center2, context1], [center2, context2]]
for i in range(C, len(word_list)-C):
    center_idx = word2idx[word_list[i]]  # center word idx
    context_word_list = word_list[i-C:i] + word_list[i+1: i+1+C]
    context_idx = [word2idx[w] for w in context_word_list]
    for c in context_idx:
        skip_gram.append([center_idx, c])



def make_data(skip_gram):
    input_data = []
    target_data = []
    for i in range(len(skip_gram)):   # 这里需要将input编码为one-hot形式
        input = np.eye(vocab_size)[skip_gram[i][0]]
        target = skip_gram[i][1]

        input_data.append(input)
        target_data.append(target)
    return input_data, target_data

input_data, target_data = make_data(skip_gram)
input_data, target_data = torch.FloatTensor(input_data), torch.LongTensor(target_data)
dataset = Data.TensorDataset(input_data, target_data)
loader = Data.DataLoader(dataset, batch_size, True)


class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        # W and V is not Transpose relationship
        self.W = nn.Parameter(torch.randn(vocab_size, embed_size).type(dtype))
        self.V = nn.Parameter(torch.randn(embed_size, vocab_size).type(dtype))


    def forward(self, X):
        """
        :param X: [batch_size, vocab_size]
        :return: output [batch_size, vocab_size]
        there is no need for activate function
        """
        hidden = torch.matmul(X, self.W)
        output = torch.matmul(hidden, self.V)
        return output


model = Word2Vec().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# training
for epoch in range(5000):
    for i, (x_batch, y_batch) in enumerate(loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        if (epoch+1)%1000 == 0:
            print(epoch+1, i, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for i, label in enumerate(vocab):
  W, WT = model.parameters()
  x,y = float(W[i][0]), float(W[i][1])
  plt.scatter(x, y)
  plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()







