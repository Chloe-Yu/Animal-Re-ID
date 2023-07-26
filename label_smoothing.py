import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, joint_all=False, num_other=0):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.joint_all = joint_all
        self.num_other = num_other

    def forward(self, x, target):
        # smoothing = torch.Tensor(np.random.uniform(0., self.smoothing, size=[x.size(0)])).cuda()
        # confidence = 1. - smoothing
        if self.joint_all:
            positions = [i for i in range(x.size(0)) if (i%2==0) or i>=2*self.num_other]
            x = x[torch.tensor(positions).cuda()]
            target = target[torch.tensor(positions).cuda()]
            
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
if __name__ == '__main__':
    loss = LabelSmoothingCrossEntropy(smoothing=0.1,joint_all=True,num_other=8)
    a = Variable(torch.rand(32, 10))
    print('a',a)
    label = Variable(torch.ones((32,)).long()) 
    print(loss(a, label))