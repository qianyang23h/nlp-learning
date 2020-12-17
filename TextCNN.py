import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.FloatTensor


# 3 words sentences (=sequence_length is 3)
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]

word_list = " ".join(sentences).split()
vocab = list(set(word_list))
word2idx = {w: i for i, w in enumerate(vocab)}

# parameters
vocab_size = len(vocab)
embed_size = 2
sequence_length = len(sentences[0])
n_class = len(set(labels))
batch_size = 3


def make_data(sentences, labels):
    input_data = []
    target_data = []
    for sen in sentences:
        tokens = sen.split()
        input = [word2idx[w] for w in tokens]
        input_data.append(input)

    for label in labels:
        target_data.append(label)

    return input_data, target_data   # input_data: [[1,2],[3,1],...]


input_data, target_data = make_data(sentences, labels)
input_data, target_data = torch.LongTensor(input_data), torch.LongTensor(target_data)
dataset = Data.TensorDataset(input_data, target_data)
loader = Data.DataLoader(dataset, batch_size, True)


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        output_channel = 3
        self.W = nn.Embedding(vocab_size, embed_size)
        # conv
        self.conv = nn.Sequential(
            # Args: [in_channels, out_channels, (filter_height, filter_width), stride, padding]
            nn.Conv2d(1, output_channel, (2, embed_size)),
            nn.ReLU(),
            # pool : ((filter_height, filter_width))
            nn.MaxPool2d((2, 1))  # 这里的maxpooling的形状需要根据sequence_length的变化而改变
        )
        # fc
        self.fc = nn.Linear(output_channel, n_class)


    def forward(self, X):
        # X : [batch_size, sequence_length]
        batch_size = X.shape[0]  # testset may have different batch_size, so get it by X.shape[0]
        embed_X = self.W(X)  # [batch_size, sequence_length, embed_size]
        embed_X = embed_X.unsqueeze(1)  # add channel [batch_size, channel(=1), sequence_length, embed_size]
        conved = self.conv(embed_X)  # [batch_size, output_channel, 1, 1]
        flatten = conved.view(batch_size, -1)  # [batch_size, output_channel*1*1]
        output = self.fc(flatten)
        return output


model = TextCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)



for epoch in range(5000):
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# testing
test_text = 'i hate me'
tests = [[word2idx[n] for n in test_text.split()]]
test_batch = torch.LongTensor(tests).to(device)
model = model.eval()
predict = model(test_batch)  # [batch_size, n_class]
predict_index = predict.max(1)[1]
if predict[0][0] == 0:
    print(test_text, "is Bad Mean...")
else:
    print(test_text, "is Good Mean!!")











