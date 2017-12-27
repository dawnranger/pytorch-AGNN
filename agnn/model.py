import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(self, requires_grad=True):
        super(GraphAttentionLayer, self).__init__()
        if requires_grad:
            # unifrom initialization
            self.beta = Parameter(torch.Tensor(1).uniform_(
                0, 1), requires_grad=requires_grad)
        else:
            self.beta = Variable(torch.zeros(1), requires_grad=requires_grad)

    def forward(self, x, adj):
        # NaN grad bug fixed at pytorch 0.3. Release note:
        #     `when torch.norm returned 0.0, the gradient was NaN.
        #     We now use the subgradient at 0.0, so the gradient is 0.0.`
        norm2 = torch.norm(x, 2, 1).view(-1, 1)

        # add a minor constant (1e-7) to denominator to prevent division by
        # zero error
        cos = self.beta * \
            torch.div(torch.mm(x, x.t()), torch.mm(norm2, norm2.t()) + 1e-7)

        # neighborhood masking (inspired by this repo:
        # https://github.com/danielegrattarola/keras-gat)
        mask = (1. - adj) * -1e9
        masked = cos + mask

        # propagation matrix
        P = F.softmax(masked, dim=1)

        # attention-guided propagation
        output = torch.mm(P, x)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (16 -> 16)'


class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features, initializer=nn.init.xavier_uniform):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(initializer(
            torch.Tensor(in_features, out_features)))

    def forward(self, input):
        # no bias
        return torch.mm(input, self.weight)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class AGNN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers, dropout_rate):
        super(AGNN, self).__init__()

        self.layers = nlayers
        self.dropout_rate = dropout_rate

        self.embeddinglayer = LinearLayer(nfeat, nhid)
        nn.init.xavier_uniform(self.embeddinglayer.weight)

        self.attentionlayers = nn.ModuleList()
        # for Cora dataset, the first propagation layer is non-trainable
        # and beta is fixed at 0
        self.attentionlayers.append(GraphAttentionLayer(requires_grad=False))
        for i in range(1, self.layers):
            self.attentionlayers.append(GraphAttentionLayer())

        self.outputlayer = LinearLayer(nhid, nclass)
        nn.init.xavier_uniform(self.outputlayer.weight)

    def forward(self, x, adj):
        x = F.relu(self.embeddinglayer(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)

        for i in range(self.layers):
            x = self.attentionlayers[i](x, adj)

        x = self.outputlayer(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        return F.log_softmax(x, dim=1)
