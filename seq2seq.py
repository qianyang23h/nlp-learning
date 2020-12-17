import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# ?: Symbol that will fill in blank sequence if current batch data size is short than n_step
# n_step: save the length of the longest word


# char级别的分词
letter = [c for c in "SE?abcdefghijklmnopqrstuvwxyz"]
letter2idx = {w: i for i, w in enumerate(letter)}
seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

# seq2seq parameter
n_step = max([max(len(i), len(j)) for i, j in seq_data])
n_hidden = 128
n_class = len(letter2idx)
batch_size = 3


# data processing

def make_data(seq_data):
    """
    :param seq_data:
    :return:
        enc_input_all: [6, n_step+1 (because of 'E'), n_class]
        dec_input_all: [6, n_step+1 (because of 'S'), n_class]
        dec_output_all: [6, n_step+1 (because of 'E')]
    """
    enc_input_all, dec_input_all, dec_output_all = [], [], []
    for seq in seq_data:
        for i in range(2):
            # 单词长度不够，用"?"填充
            seq[i] = seq[i] + "?" * (n_step - len(seq[i]))   # ['man??', 'women']

        enc_input = [letter2idx[w] for w in (seq[0] + "E")]   # Encoder输入数据末尾添加终止符"E"
        dec_input = [letter2idx[w] for w in ("S" + seq[1])]   # Decoder输入数据开头添加开始符"S"
        dec_output = [letter2idx[w] for w in (seq[1] + "E")]  # Decoder 的输出数据末尾添加结束标志 'E'

        enc_input_all.append(np.eye(n_class)[enc_input])
        dec_input_all.append(np.eye(n_class)[dec_input])
        dec_output_all.append(dec_output)  # not one-hot

    # make tensor
    return torch.Tensor(enc_input_all), torch.Tensor(dec_input_all), torch.LongTensor(dec_output_all)

enc_input_all, dec_input_all, dec_output_all = make_data(seq_data)


# 由于有三个数据要返回，所有要重写DataSet类
# 具体来说就是继承 torch.utils.data.Dataset 类，然后实现里面的__len__以及__getitem__方法
class TranslateDataSet(Data.Dataset):
    def __init__(self, enc_input_all, dec_input_all, dec_output_all):
        self.enc_input_all = enc_input_all
        self.dec_input_all = dec_input_all
        self.dec_output_all = dec_output_all

    def __len__(self): # return dataset size
        return len(enc_input_all)

    def __getitem__(self, idx):
        return self.enc_input_all[idx], self.dec_input_all[idx], self.dec_output_all[idx]


loader = Data.DataLoader(TranslateDataSet(enc_input_all, dec_input_all, dec_output_all), batch_size, True)


# model
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5, bidirectional=False)
        self.decoder = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5, bidirectional=False)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_h_0, dec_input):
        # enc_input: [batch_size, n_step+1, n_class]
        # dec_input: [batch_size, n_step+1, n_class]
        enc_input = enc_input.transpose(0, 1)  # [n_step+1, batch_size, n_class]
        dec_input = dec_input.transpose(0, 1)  # [n_step+1, batch_size, n_class]

        _, h_t = self.encoder(enc_input, enc_h_0)  # h_t: [num_layer(=1)*num_directions(=1), batch_size, n_hidden]
        output, _ = self.decoder(dec_input, h_t)   # output: [n_step+1, batch_size, num_directions(=1)*n_hidden(=128)]
        output = self.fc(output)  # output: [n_step+1, batch_size, n_class]
        return output

torch.cat()
model = Seq2Seq().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(5000):
    for enc_input_batch, dec_input_batch, dec_output_batch in loader:
        # enc_input_all: [batch_size, n_step + 1, n_class]
        # dec_input_all: [batch_size, n_step + 1, n_class]
        # dec_output_all: [batch_size, n_step + 1]
        (enc_input_batch, dec_input_batch, dec_output_batch) = (
            enc_input_batch.to(device), dec_input_batch.to(device), dec_output_batch.to(device))

        # h_0: [num_layer(=1)*num_direction, batch_size, n_hidden]
        h_0 = torch.randn(1, batch_size, n_hidden).to(device)
        # pred: [n_step+1, batch_size, n_class]
        pred = model(enc_input_batch, h_0, dec_input_batch)

        # 因为这里的pred是一个三维结果，所以我们这里需要一个循环叠加loss
        pred = pred.transpose(0, 1)  # [batch_size, n_step+1(=6), n_class]
        loss = 0
        for i in range(len(dec_output_batch)):
            # pred[i] : [n_step+1, n_class]
            # dec_output_batch[i] : [n_step+1]
            loss += criterion(pred[i], dec_output_batch[i])
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Testing
def translate(word):
    enc_input, dec_input, _ = make_data([[word, '?' * n_step]])
    enc_input, dec_input = enc_input.to(device), dec_input.to(device)
    # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
    hidden = torch.zeros(1, 1, n_hidden).to(device)
    output = model(enc_input, hidden, dec_input)
    # output : [n_step+1, batch_size, n_class]

    predict = output.data.max(2, keepdim=True)[1] # select n_class dimension
    decoded = [letter[i] for i in predict]
    translated = ''.join(decoded[:decoded.index('E')])

    return translated.replace('?', '')

print('test')
print('man ->', translate('man'))
print('mans ->', translate('mans'))
print('king ->', translate('king'))
print('black ->', translate('black'))
print('up ->', translate('up'))





