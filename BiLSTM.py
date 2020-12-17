import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

dtype = torch.FloatTensor

sentence = (
    'GitHub Actions makes it easy to automate all your software workflows '
    'from continuous integration and delivery to issue triage and more'
)

word2idx = {w: i for i, w in enumerate(list(set(sentence.split())))}
idx2word = {i: w for i, w in enumerate(list(set(sentence.split())))}

# BiLSTM parameters
n_class = len(word2idx)
n_hidden = 5
max_len = len(sentence.split())
batch_size = 16


def make_data(sentence):
    input_data = []
    target_data = []

    for i in range(max_len-1): # 这里的batch_size=max_len-1
        tokens = sentence.split()
        input = [word2idx[w] for w in tokens[:i+1]]
        input = input + [0] * (max_len - i - 1)   # 用0补全
        input = np.eye(n_class)[input]
        target = word2idx[tokens[i+1]]

        input_data.append(input)
        target_data.append(target)

    return input_data, target_data  # input_data: [max_len-1, max_len, n_class]


input_data, target_data = make_data(sentence)
input_data, target_data = torch.Tensor(input_data), torch.LongTensor(target_data)
dataset = Data.TensorDataset(input_data, target_data)
loader = Data.DataLoader(dataset, 16, True)


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        self.fc = nn.Linear(n_hidden * 2, n_class)   # 因为是双向，所以需要乘以2


    def forward(self, X):
        # X： [batch_size, max_len, n_class]
        batch_size = X.shape[0]
        X = X.transpose(0, 1)  # X : [max_len, batch_size, n_class]

        # h_0 and c_0 shape:[num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        # If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.
        h_0 = torch.randn(1*2, batch_size, n_hidden)
        c_0 = torch.randn(1*2, batch_size, n_hidden)

        output, (_, _) = self.lstm(X, (h_0, c_0))
        output = output[-1]  # 取最后一个时间步的输出 outputs ：[batch_size, n_hidden * 2]
        output = self.fc(output)
        return output

model = BiLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training
for epoch in range(10000):
    for x, y in loader:
      pred = model(x)
      loss = criterion(pred, y)
      if (epoch + 1) % 1000 == 0:
          print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

# Pred
predict = model(input_data).data.max(1, keepdim=True)[1]
print(sentence)
print([idx2word[n.item()] for n in predict.squeeze()])