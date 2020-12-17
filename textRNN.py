import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sentences = ["i like cat", "i love Riemann", "i hate milk"]
word_list = " ".join(sentences).split()
vocab = list(set(word_list))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
n_class = len(vocab)

# textRNN parameters
n_hidden = 5
n_step = 2  # number of step(= number of input words)
batch_size = 2


def make_data(sentences):
    # input use one-hot encoding
    input_data = []
    target_data = []

    for sen in sentences:
        tokens = sen.split()
        input = [word2idx[w] for w in tokens[:-1]]
        target = word2idx[tokens[-1]]

        # 将input编码为one-hot
        input = np.eye(n_class)[input]
        input_data.append(input)   # input_data size [batch_size, n_step, n_class]
        target_data.append(target)

    return input_data, target_data

input_batch, target_batch = make_data(sentences)
input_batch, target_batch = torch.Tensor(input_batch), torch.LongTensor(target_batch)
dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset, batch_size, True)


class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        # rnn Args: (input_size, hidden_size, num_layers, bidirectional)
        # rnn inputs: input:(seq_len, batch, input_size)  h_0:(num_layers * num_directions, batch, hidden_size)
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden, num_layers=1, bidirectional=False)
        # fc
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, X, h_0):
        # X [batch_size, n_step, n_class]
        X = X.transpose(0,1)  # [n_step, batch_size, n_class]
        out, hidden = self.rnn(X, h_0)
        # out: [n_step, batch_size, num_directions(=1)*n_hidden ]
        # hidden: [num_layers * num_directions, batch_size, hidden_size]
        out = out[-1]  # 取最后一个时间步的输出  size：[batch_size, num_directions(=1) * n_hidden]
        out = self.fc(out)
        return out

model = TextRNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5000):
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        h_0 = torch.randn(1, x_batch.shape[0], n_hidden).to(device)
        # h_0 = None
        pred = model(x_batch, h_0)
        loss = criterion(pred, y_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

input = [sen.split()[:2] for sen in sentences]
# Predict
hidden = torch.zeros(1, len(input), n_hidden).to(device)
input_batch = input_batch.to(device)
predict = model(input_batch, hidden).data.max(1, keepdim=True)[1]
print([sen.split()[:2] for sen in sentences], '->', [idx2word[n.item()] for n in predict.squeeze()])





