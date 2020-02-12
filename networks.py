import torch.nn as nn
import torch.nn.functional as F
import torch


class EmbeddingNetX(nn.Module):
    def __init__(self):
        super(EmbeddingNetX, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(), nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(), nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256), nn.PReLU(), nn.Linear(256, 256),
                                nn.PReLU(), nn.Linear(256, 32))

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNetX(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNetX, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(32, n_classes)

        self.weight = torch.nn.Parameter(torch.rand(n_classes, 32))

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)

        #         scores = (output.unsqueeze(1) - self.weight).norm(dim=-1) * -1
        #         scores = F.log_softmax(output, dim=-1)

        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
