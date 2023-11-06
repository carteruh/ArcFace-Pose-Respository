import torch
import math
import torch.nn.functional as F


class ArcFaceLoss(torch.nn.Module):
    """Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)
        self.device= torch.device("cuda")

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, feature, label= None):
        # Normalize embeddings and weights
        feature = F.normalize(feature)
        weights = F.normalize(self.weight)

        # Compute the logit
        cos_theta = F.linear(feature, weights)
        
        if (label is None):
            return cos_theta

        # Find the angle between the weight and feature
        theta = torch.acos(cos_theta)

        # Add angular margin penalty
        marginal_target_logit = torch.cos(theta + self.m)
        
        marginal_target_logit = torch.where(cos_theta > self.th, marginal_target_logit, cos_theta - self.mm)

        # One-hot encoding
        one_hot = torch.zeros(cos_theta.size(), device=self.device)
       # one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Compute class wise similarity score through element wise multiplication of one_hot ground truth and the marginal target logit
        score = (one_hot * marginal_target_logit) + ((1.0 - one_hot) * cos_theta)
        #score = torch.mul(marginal_target_logit, one_hot)

        # Rescale to s
        score *= self.s

        return score