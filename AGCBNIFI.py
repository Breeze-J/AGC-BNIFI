import torch
from torch import nn
from torch.nn import Parameter
from AE import AE
from GNNLayer import GNNLayer
from opt import args

device = torch.device("cuda" if args.cuda else "cpu")

class AttentionLayer(nn.Module):
    def __init__(self, last_dim, n_num):
        super(AttentionLayer, self).__init__()
        self.n_num = n_num
        self.fc1 = nn.Linear(n_num * last_dim, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, n_num)
        self.attention = nn.Softmax(dim=1)
        self.relu = nn.LeakyReLU()
        self.T = 1
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        attention_sample = self.attention(x / self.T)
        attention_view = torch.mean(attention_sample, dim=0, keepdim=True).squeeze()
        return attention_view

class FusionLayer(nn.Module):
    def __init__(self, last_dim, n_num=2):
        super(FusionLayer, self).__init__()
        self.n_num = n_num
        self.attentionLayer = AttentionLayer(last_dim, n_num)
    def forward(self, x, k):
        y = torch.cat((x, k), 1)
        weights = self.attentionLayer(y)
        x_TMP = weights[0] * x + weights[1] * k
        return x_TMP


class AGCBNIFI(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3,
                 ae_n_dec_1, ae_n_dec_2, ae_n_dec_3,
                 gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,
                 gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,
                 n_input, n_z, n_clusters, v):
        super(AGCBNIFI, self).__init__()

        self.ae = AE(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)

        ae_pre = 'ae_pretrain/{}.pkl'.format(args.name)
        self.ae.load_state_dict(torch.load(ae_pre, map_location='cpu'))
        print('Loading AE pretrain model:', ae_pre)

        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)
        self.gnn_4 = GNNLayer(gae_n_enc_3, n_z)

        self.gnn_5 = GNNLayer(n_z, gae_n_dec_1)
        self.gnn_6 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_7 = GNNLayer(gae_n_dec_2, gae_n_dec_3)
        self.gnn_8 = GNNLayer(gae_n_dec_3, n_input)

        self.fuse1 = FusionLayer(gae_n_enc_1)
        self.fuse2 = FusionLayer(gae_n_enc_2)
        self.fuse3 = FusionLayer(gae_n_enc_3)
        self.fuse4 = FusionLayer(n_z)
        self.s = nn.Sigmoid()

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = v
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, adj):


        z_ae, x_bar, enc_h1, enc_h2, enc_h3, dec_h1, dec_h2, dec_h3 = self.ae(x)


        gae_enc1 = self.gnn_1(x, adj, active=True)
        h = self.fuse1(gae_enc1, enc_h1)
        gae_enc2 = self.gnn_2(h, adj, active=True)
        h = self.fuse2(gae_enc2, enc_h2)
        gae_enc3 = self.gnn_3(h, adj, active=True)
        h = self.fuse3(gae_enc3, enc_h3)
        z_gae = self.gnn_4(h, adj, active=False)

        z_i = self.fuse4(z_gae, z_ae)
        z_l = torch.spmm(adj, z_i)

        gae_dec1 = self.gnn_5(z_gae, adj, active=True)
        gae_dec2 = self.gnn_6(gae_dec1, adj, active=True)
        gae_dec3 = self.gnn_7(gae_dec2, adj, active=True)
        z_hat = self.gnn_8(gae_dec3, adj, active=True)

        adj_hat = self.s(torch.mm(z_hat, z_hat.t()))

        q = 1.0 / (1.0 + torch.sum(torch.pow(z_l.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        return x_bar, z_hat, adj_hat, z_ae, q, q1, z_l


