import torchtext.data as data
import torchtext.datasets as datasets
from torchtext.vocab import GloVe
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch
from torch.autograd import Variable
# from imdb import IMDB


class EmbNet(nn.Module):
    def __init__(self, emb_size, hidden_size1, hidden_size2):
        super(EmbNet, self).__init__()
        self.embedding = nn.Embedding(emb_size, hidden_size1)
        self.fc = nn.Linear(hidden_size2, 3)
    def forward(self, x):
        embeds = self.embedding(x).view(x.size(0), -1)
        out = self.fc(embeds)
        return F.log_softmax(out, dim=-1)


class Rnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Rnn, self).__init__()
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
        combine = torch.cat((input, hidden), 1)
        hidden_out = self.i2h(combine)
        output = F.log_softmax(self.i2o(combine), dim=1)
        return hidden_out, output


class LstmImdb(nn.Module):
    def __init__(self, vocab, hidden_size, num_layer, classes):
        super(LstmImdb, self).__init__()
        self.embed = nn.Embedding(vocab, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layer)
        self.fc = nn.Linear(hidden_size, classes)
        self.soft = nn.LogSoftmax(dim=-1)
        self.nl = num_layer
        self.hidden_size = hidden_size

    def forward(self, x):
        embedding = self.embed(x)
        batch_size = x.size()[1]
        h0 = c0 = Variable(embedding.data.new(*(self.nl, batch_size, self.hidden_size)).zero_())
        out, _ = self.lstm(embedding, (h0, c0))
        out = out[-1]
        out = self.fc(out)
        # out = F.dropout(out)
        out = self.soft(out)
        return out


class ConvOned(nn.Module):
    def __init__(self, vocab, hidden_size, classes, kernal_size, max_length):
        super(ConvOned, self).__init__()
        self.embed = nn.Sequential(
            nn.Embedding(vocab, hidden_size),
            nn.Conv1d(max_length, hidden_size, kernal_size),
            nn.AdaptiveAvgPool1d(10),
        )
        self.fc = nn.Sequential(
            nn.Linear(1000, classes),
            # nn.Dropout(),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.fc(x.view(x.size()[0], -1))
        return x


def fit(epoch, model, data_loader, mode, is_cuda, optim):
    if mode == 'train':
        model = model.train()
    if mode == 'val':
        model = model.eval()
    running_loss = 0
    running_correct = 0
    for batch_idx, data_batch in enumerate(data_loader):
        data, label = data_batch.text, data_batch.label
        if is_cuda:
            model, data, label = model.cuda(), data.cuda(), label.cuda()
        if mode == 'train':
            optim.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, label)
        if mode == 'train':
            loss.backward()
            optim.step()
        running_loss += loss.item()
        pre = output.max(dim=1, keepdim=True)[1]
        equal = pre.eq(label.view_as(pre)).cpu().sum()
        running_correct += equal.item()
    average_loss = running_loss / len(data_loader.dataset)
    accuracy = running_correct / len(data_loader.dataset)
    print("mode:%s, epoch:%d, loss:%f, acc:%f" %(mode, epoch, average_loss, accuracy))
    return average_loss, accuracy


def main():
    Text = data.Field(lower=True, batch_first=True, fix_length=200)
    Label = data.Field(sequential=False)

    train, test = datasets.IMDB.splits(Text, Label)
    print(len(train))
    Text.build_vocab(train, vectors=GloVe(name='6B', dim=300), max_size=10000, min_freq=10)
    Label.build_vocab(train)

    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=32, device='cpu', shuffle=True)
    train_iter.repeat = False
    test_iter.repeat = False
    # **********************embnet***************************
    # model = EmbNet(len(Text.vocab.stoi), 300, 12000)
    # model.embedding.weight.data = Text.vocab.vectors
    # model.embedding.weight.requires_grad = False
    # optimizer = opt.Adam([param for param in model.parameters() if param.requires_grad == True], lr=0.001)
    # **********************LSTM******************************
    # model = LstmImdb(len(Text.vocab.stoi), 100, 2, 3)
    # optimizer = opt.Adam(model.parameters(), lr=1e-3)
    # **********************conv1d****************************
    model = ConvOned(len(Text.vocab.stoi), 100, 3, 3, 200)
    optimizer = opt.Adam(model.parameters(), lr=1e-3)

    is_cuda = torch.cuda.is_available()
    #**************train******************
    epoch = 20
    for i in range(epoch):
        fit(i, model, train_iter, "train", is_cuda, optimizer)
        fit(i, model, test_iter, "val", is_cuda, optimizer)

if __name__ == "__main__":
    main()
