import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import numpy as np

def loss_kl(mu, logvar, nodemask=None,reduction='mean'):
    if reduction=='mean':
        f=torch.mean
        if nodemask is None:
            s=mu.size()[0]
        else:
            s=nodemask.size()[0]
    elif reduction=='sum':
        f=torch.sum
        s=1
    if nodemask is None:
        kl= -(0.5 / s) * f(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return kl
    kl= -(0.5 / s) * f(torch.sum(1 + 2 * logvar[nodemask] - mu[nodemask].pow(2) - logvar[nodemask].exp().pow(2), 1))
    return kl

# def loss_nb(preds,y_true,mask,reconWeight,eps = 1e-10,ifmean=True):
#     #adapted from https://github.com/theislab/dca/blob/master/dca/loss.py
#     output,pi,theta,y_pred=preds
#     nbloss1=torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
#     nbloss2=(theta+y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))
#     nbloss=nbloss1+nbloss2
    
# #     nbloss=torch.where(torch.isnan(nbloss), torch.zeros_like(nbloss)+np.inf, nbloss)
#     if ifmean:
#         return torch.mean(nbloss[mask])*reconWeight
#     else:
#         return nbloss
    
def loss_nb(preds,y_true,reconWeight,eps = 1e-10,ifmean=True):
    #adapted from https://github.com/theislab/dca/blob/master/dca/loss.py
    output,pi,theta,y_pred=preds
    nbloss1=torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
    nbloss2=(theta+y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))
    nbloss=nbloss1+nbloss2
    
#     nbloss=torch.where(torch.isnan(nbloss), torch.zeros_like(nbloss)+np.inf, nbloss)
    if ifmean:
        return torch.mean(nbloss)*reconWeight
    else:
        return nbloss
    
def loss_zinb(preds,y_true,reconWeight,ridgePi,eps = 1e-10):
    output,pi,theta,y_pred=preds
    nb_case=loss_nb(preds,y_true,reconWeight,eps = 1e-10,ifmean=False)- torch.log(1.0-pi+eps)
    zero_nb = torch.pow(theta/(theta+y_pred+eps), theta)
    zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
    result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
    ridge = ridgePi*pi*pi
    result += ridge
    result=torch.mean(result)

    return result*reconWeight



# def loss_CE(preds, labels, pos_weight, norm,nodemask=None):
#     if nodemask is None:
#         cost=norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight,reduction='mean')
#         return cost
#     cost=norm * F.binary_cross_entropy_with_logits(preds[nodemask,:][:,nodemask], labels[nodemask,:][:,nodemask], pos_weight=pos_weight,reduction='mean')
#     return cost

def loss_CE(preds, labels, pos_weight, norm,nodemask=None):
    if nodemask is None:
        cost=norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight,reduction='mean')
        return cost
    cost=norm * F.binary_cross_entropy_with_logits(preds[nodemask,:][:,nodemask], labels[nodemask,:][:,nodemask], pos_weight=pos_weight,reduction='mean')
    return cost

def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


def Noise_Cross_Entropy(pred_sp, emb_sp,graph_neigh):
    '''\
    Calculate noise cross entropy. Considering spatial neighbors as positive pairs for each spot

    Parameters
    ----------
    pred_sp : torch tensor
        Predicted spatial gene expression matrix.
    emb_sp : torch tensor
        Reconstructed spatial gene expression matrix.

    Returns
    -------
    loss : float
        Loss value.

    '''

    mat = cosine_similarity(pred_sp, emb_sp) 
    k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))

    # positive pairs
    p = torch.exp(mat)
    p = torch.mul(p,graph_neigh).sum(axis=1)

    ave = torch.div(p, k)
    loss = - torch.log(ave).mean()

    return loss

def cosine_similarity(pred_sp, emb_sp):  #pres_sp: spot x gene; emb_sp: spot x gene
    '''\
    Calculate cosine similarity based on predicted and reconstructed gene expression matrix.    
    '''

    M = torch.matmul(pred_sp, emb_sp.T)
    Norm_c = torch.norm(pred_sp, p=2, dim=1)
    Norm_s = torch.norm(emb_sp, p=2, dim=1)
    Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
    M = torch.div(M, Norm)

    if torch.any(torch.isnan(M)):
       M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

    return M     