import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

dtype = torch.FloatTensor


# data preprocessing
sentences = ['i like Riemann', 'i love Pigcat', 'i miss Manman']
word_list = " ".join(sentences).split(" ")  # ['i', 'like', 'Riemann' ,'i', 'love',...]
vocab = list(set(word_list))
word2idx_dict = {w: i for i, w in enumerate(vocab)}
idx2word_dict = {i: w for i, w in enumerate(vocab)}


# parameter
n_step = len(sentences[0].split())-1  # use n_step word as the input word
n_hidden = 2
embed_size = 2
n_class = len(vocab)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_data(sentences):
    input_data = []
    target_data = []

    for sen in sentences:
        word = sen.split()  # ['i', 'like', 'Riemann']
        input = [word2idx_dict[w] for w in word[:-1]]
        target = word2idx_dict[word[-1]]

        input_data.append(input)
        target_data.append((target))

    return input_data, target_data


input_data, target_data = make_data(sentences)
input_data = torch.LongTensor(input_data)     # input_data：[[1,4],[1,2],[1,0]]
target_data = torch.LongTensor(target_data)   # target_data: [3,0,3]
dataset = Data.TensorDataset(input_data, target_data)
loader = Data.DataLoader(dataset, batch_size=16, shuffle=True)


class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, embed_size)
        self.H = nn.Parameter(torch.randn(n_step*embed_size, n_hidden).type(dtype))
        self.W = nn.Parameter(torch.randn(n_step*embed_size, n_class).type(dtype))
        self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
        self.b_1 = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.b_2 = nn.Parameter(torch.randn(n_class).type(dtype))

    def forward(self, X):
        X = self.C(X)  # [batch_size, n_step] --> [batch_size, n_step, embed_size]
        X = X.view(-1, n_step*embed_size)  # [batch_size, n_step*embed_size]
        hidden_out = torch.tanh(torch.mm(X, self.H) + self.b_1)  # [batch_size, n_hidden]
        output = torch.mm(X, self.W) + torch.mm(hidden_out, self.U)  # [batch_size, n_class]

        return output


model = NNLM().to(device)  # model
criterion = nn.CrossEntropyLoss()  # loss function
optimizer = optim.Adam(model.parameters(), lr=1e-1)


# training
for epoch in range(5000):
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print("epoch: %04d" %(epoch+1), "loss : %.8f" %loss.item())  # loss是tensor格式需要通过item()转换为普通python类型


# test
output = model(input_data).max(1, keepdim=False)[1]
print(output)
print([idx2word_dict[i.item()] for i in output])   # 这里需要使用item()将tensor






