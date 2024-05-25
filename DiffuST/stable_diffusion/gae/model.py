import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")#this should be set to the GPU device you would like to use on your machine

class GraphConvolution(nn.Module):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout=0., act=F.relu,bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.weight=nn.parameter.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        initialization.
        """
        # nn.init.xavier_uniform_(self.weight)
        # nn.init.xavier_uniform_(self.bias)
        # nn.init.kaiming_uniform_(self.weight)
        # nn.init.kaiming_uniform_(self.bias)
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     nn.init.uniform_(self.bias, -bound, bound)
        init_range = np.sqrt(6.0 / (self.input_dim + self.output_dim))
        nn.init.uniform_(self.weight, a=-init_range, b=init_range)
        
        if self.bias is not None:
            nn.init.uniform_(self.bias,a=-init_range, b=init_range)         
        
        
    def forward(self,feat,adj):
        feat = F.dropout(feat, self.dropout, self.training)
        support = torch.mm(feat, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            output= output + self.bias
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> '+ str(self.output_dim) + ')'
    
class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self, dropout, act=lambda x: x):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, inputs):
        inputs=F.dropout(inputs, self.dropout, training=self.training)
        x=torch.mm(inputs, inputs.t())
        outputs = self.act(x)
        return outputs


class FC(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0., act=F.leaky_relu, batchnorm=False,bias=False):
        super(FC, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.batchnorm=batchnorm
        self.linearlayer=nn.Linear(input_dim,output_dim,bias=bias)
        if batchnorm:
            self.batchnormlayer=nn.BatchNorm1d(output_dim)
    
    def forward(self,inputs):
        inputs = F.dropout(inputs, self.dropout, self.training)
        output = self.linearlayer(inputs)
        if self.batchnorm:
            output=self.batchnormlayer(output)
        output = self.act(output)
        return output    
    


class STACI(nn.Module):   
    def __init__(self, input_feat_dim, hidden_dim1,hidden_dim2,hidden_decoder, dropout,meanMin=1e-5,meanMax=1e6,thetaMin=1e-5,thetaMax=1e6):
        super(STACI, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.leaky_relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.gc2s = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.fc1 = FC(hidden_dim2, hidden_decoder, dropout, act = F.leaky_relu, batchnorm = True)
        self.pi=FC(hidden_decoder, input_feat_dim, dropout=0, act = torch.sigmoid, batchnorm = False,bias=True)
        self.theta=FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(F.softplus(x),min=thetaMin,max=thetaMax), batchnorm = False,bias=True)
        self.mean=FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(torch.exp(x),min=meanMin,max=meanMax), batchnorm = False,bias=True)


    def encode(self, x, adj):
        hidden1=self.gc1(x,adj)
        return self.gc2(hidden1, adj), self.gc2s(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        output = self.fc1(z)
        pi_res=self.pi(output)
        theta_res=self.theta(output)
        mean_res=self.mean(output)
        return output,pi_res,theta_res,mean_res
    
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        #output adj_recon,mu,logvar,z,features_recon
        return self.dc(z), mu, logvar, z, self.decode_X(z)
    
class GCNModelVAE_FC(nn.Module):   
    def __init__(self, in_features, hidden_dim, out_features, hidden_decoder, dropout, meanMin=1e-5, meanMax=1e6, thetaMin=1e-5, thetaMax=1e6):
        super(GCNModelVAE_FC, self).__init__()
        self.encoder = Encoder(in_features, hidden_dim, out_features, dropout)      
        self.decoder = Decoder(in_features, out_features, hidden_decoder, dropout,meanMin, meanMax, thetaMin, thetaMax) 
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu    
    
    def forward(self, x, adj):
        mu, logvar = self.encoder(x, adj)
        z = self.reparameterize(mu, logvar)
        #output adj_recon,mu,logvar,z,features_recon
        return self.dc(z), mu, logvar, z, self.decoder(z,adj)
    
    
class Encoder(nn.Module):
    def __init__(self, in_features, hidden_dim,out_features, dropout):
        super(Encoder, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden_dim, dropout, act=F.leaky_relu)
        self.gc2 = GraphConvolution(hidden_dim, out_features, dropout, act=F.leaky_relu)
        self.gc2s = GraphConvolution(hidden_dim, out_features, dropout, act=F.leaky_relu) 
    
    def forward(self, x, adj):
        hidden1=self.gc1(x,adj)
        return self.gc2(hidden1, adj), self.gc2s(hidden1, adj)
    
class Decoder(nn.Module):
    def __init__(self, in_features, out_features, hidden_decoder, dropout,meanMin, meanMax, thetaMin, thetaMax):
        super(Decoder, self).__init__()
        self.fc1 = GraphConvolution(out_features, hidden_decoder, dropout, act = F.leaky_relu)        
        self.pi=GraphConvolution(hidden_decoder, in_features, dropout=0, act = torch.sigmoid)
        self.theta=GraphConvolution(hidden_decoder, in_features, dropout=0, act = lambda x: torch.clamp(F.softplus(x),min=thetaMin,max=thetaMax))
        self.mean=GraphConvolution(hidden_decoder, in_features, dropout=0, act = lambda x: torch.clamp(torch.exp(x),min=meanMin,max=meanMax))   
        # self.dec=GraphConvolution(hidden_decoder, in_features, dropout=0, act = F.leaky_relu)   
        # self.pi=FC(hidden_decoder, in_features, dropout=0, act = torch.sigmoid, batchnorm = False,bias=True)
        # self.theta=FC(hidden_decoder, in_features, dropout=0, act = lambda x: torch.clamp(F.softplus(x),min=thetaMin,max=thetaMax), batchnorm = False,bias=True)
        # self.mean=FC(hidden_decoder, in_features, dropout=0, act = lambda x: torch.clamp(torch.exp(x),min=meanMin,max=meanMax), batchnorm = False,bias=True)
        
    def forward(self,z,adj):
        output = self.fc1(z,adj)
        pi_res=self.pi(output,adj)
        theta_res=self.theta(output,adj)
        mean_res=self.mean(output,adj)
        # dec_res=self.dec(output,adj)
        # pi_res=self.pi(output)
        # theta_res=self.theta(output)
        # mean_res=self.mean(output)
        return output,pi_res,theta_res,mean_res


#############################################
    
class GCNModelNomal(nn.Module):   
    def __init__(self, in_features, hidden_dim, out_features, dropout):
        super(GCNModelNomal, self).__init__() 
        self.gc1 = GraphConvolution(in_features, hidden_dim, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim, out_features, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim, out_features, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar,z

    
class GCNModelVAE_IMG(nn.Module):   
    def __init__(self, in_features, hidden_dim, out_features,outputshape, dropout, meanMin=1e-5, meanMax=1e6, thetaMin=1e-5, thetaMax=1e6):
        super(GCNModelVAE_IMG, self).__init__()
        self.encoder = IMG_Encoder(in_features, hidden_dim, out_features, dropout)      
        self.decoder = IMG_Decoder(out_features, outputshape, dropout,meanMin, meanMax, thetaMin, thetaMax) 
        # self.dc = InnerProductDecoder(dropout, act=lambda x: x)
    
    def forward(self, x, adj):
        # mu, logvar = self.encoder(x, adj)
        # z = self.reparameterize(mu, logvar)
        #output adj_recon,mu,logvar,z,features_recon
        # return self.dc(z), mu, logvar, z, self.decoder(z,adj)
        z = self.encoder(x, adj)
        return z,self.decoder(z, adj)
        
    
class IMG_Encoder(nn.Module):
    def __init__(self, in_features, hidden_dim,out_features, dropout):
        super(IMG_Encoder, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden_dim, dropout, act=F.leaky_relu)
        self.gc2= GraphConvolution(hidden_dim, out_features, dropout, act=F.leaky_relu)
        
    
    def forward(self, x, adj):
        hidden1=self.gc1(x,adj)
        hidden2=self.gc2(hidden1, adj)
        return hidden2
    
class IMG_Decoder(nn.Module):
    def __init__(self, out_features, outputshape, dropout,meanMin, meanMax, thetaMin, thetaMax):
        super(IMG_Decoder, self).__init__()
        self.gc3= GraphConvolution(out_features, outputshape, dropout=0, act = lambda x: torch.clamp(torch.exp(x),min=meanMin,max=meanMax))
        
    def forward(self,z,adj):
        output = self.gc3(z,adj)
        return output