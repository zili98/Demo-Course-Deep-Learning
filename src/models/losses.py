from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class LogitAdjustLoss(nn.Module):
  def __init__(self, base_probs=[0.2, 0.8], tau=1.0):
    super(LogitAdjustLoss, self).__init__()
    self.base_probs = torch.tensor(base_probs)
    self.tau = torch.tensor(tau)

  def forward(self, pred, target):
    pred += torch.log(torch.Tensor(self.base_probs**self.tau + 1e-12).to(pred.device,dtype=torch.float32))
    loss = torch.nn.CrossEntropyLoss(reduction='mean')
    return loss(pred, target)

class CrossEntropyLoss(nn.Module):
  def __init__(self, weight=None, label_smoothing=0.0):
    super(CrossEntropyLoss, self).__init__()
    self.weight = weight
    self.label_smoothing = label_smoothing
    if weight is not None:
        self.weight = torch.tensor(weight,dtype=torch.float)

  def forward(self, pred, target):
    if self.weight is None:
        loss = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
    else:
        loss = torch.nn.CrossEntropyLoss(weight=self.weight, reduction='mean', label_smoothing=self.label_smoothing)
    return loss(pred, target)


class instance_weighted_loss(nn.Module):
    def __init__(self, dim=1, ):
        super(instance_weighted_loss, self).__init__()
        self.dim = dim

    def forward(self, pred, target, instance_weight):
        instance_weight = torch.tensor(instance_weight, device=pred.device)
        log_pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.type(torch.int64).data.unsqueeze(self.dim), 1)
        loss = torch.sum(-true_dist * log_pred, dim=self.dim) * instance_weight
        return torch.mean(loss)

# import torch_scatter
import numpy as np
class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_var=0.25, delta_dist=1.5, norm=2):
        super(DiscriminativeLoss, self).__init__()
        self.delta_var = float(delta_var)
        self.delta_dist = float(delta_dist)
        assert norm in [1, 2]
        self.norm=norm
        self.alpha = 1
        self.beta = 1
        self.gamma = 0.001

    def forward(self, input, target):
        """Calculate discriminative loss for instance segmentation
    
        Args:
            input ([tensor]): (bs, n_features)
            target ([tensor]): (bs,)

        Returns:
            loss: [torch.Tensor]
            loss_dict: [dict[float]]
        """
        embedding = input  #bs, h, w, ch
        target = target.unsqueeze(1) #bs, h, w, 1
        var_loss=torch.zeros(1, dtype=embedding.dtype, device=embedding.device)
        dis_loss=torch.zeros(1, dtype=embedding.dtype, device=embedding.device)
        reg_loss=torch.zeros(1, dtype=embedding.dtype, device=embedding.device)
        for emb, t in zip(embedding, target):
            #calculate cluster center
            pos_embs  =emb.float().unsqueeze(0)  #[ch]
            pos_groups = t.long()  #[1]
            
            cluster_means = torch_scatter.scatter_mean(pos_embs, pos_groups, dim=0)

            #calculate variance
            cluster_means_expanded=torch.index_select(cluster_means, dim=0, index=pos_groups) #[h*w,ch]
            variance= torch.clamp(torch.norm((pos_embs - cluster_means_expanded), self.norm, 1) -self.delta_var, min=0.0) ** 2
            variance_loss= torch_scatter.scatter_mean(variance.float(), pos_groups, dim=0) #[N_cluster]
            variance_loss=variance_loss.mean()
            
            #calculate distance
            pairwise_distance=torch.norm(cluster_means.unsqueeze(0)-cluster_means.unsqueeze(1), p=self.norm, dim=-1) #N,N
            N=len(cluster_means)
            if N==1:
                distance_loss=torch.zeros(1,dtype=embedding.dtype,device=embedding.device)
            else:
                margin = 2 * self.delta_dist * (1.0 - torch.eye(N, device=cluster_means.device, dtype=cluster_means.dtype))
                distance_loss = torch.sum(torch.clamp(margin - pairwise_distance, min=0.0) ** 2)/(2*N*(N-1))

            #calculate regularization
            mean_norm=torch.norm(cluster_means, p=self.norm, dim=1).mean()

            #sum loss
            var_loss += variance_loss
            dis_loss += distance_loss
            reg_loss += mean_norm

        bs=len(embedding)
        var_loss /= bs
        dis_loss /= bs
        reg_loss /= bs

        losses = {'VarLoss': var_loss, 'DistLoss': dis_loss, 'RegLoss': reg_loss}

        total_loss = self.alpha*var_loss + self.beta*dis_loss + self.gamma*reg_loss
        return total_loss

# class FocalLoss(nn.Module):
#     def __init__(self, class_num, alpha, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         if alpha is None:
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             self.alpha = torch.tensor(alpha, dtype=torch.float)
      
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average

#     def forward(self, inputs, targets):
#         N = inputs.size(0)
#         C = inputs.size(1)
#         P = F.softmax(inputs)

#         class_mask = inputs.data.new(N, C).fill_(0)
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)

#         if inputs.is_cuda and not self.alpha.is_cuda:
#             self.alpha = self.alpha.cuda()
#         alpha = self.alpha[ids.data.view(-1,1)]

#         probs = (P*class_mask).sum(1).view(-1,1)

#         log_p = probs.log()

#         batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss