import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        # smoothing = torch.Tensor(np.random.uniform(0., self.smoothing, size=[x.size(0)])).cuda()
        # confidence = 1. - smoothing

        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
if __name__ == '__main__':
    loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    a = Variable(torch.zeros(2, 10).cuda())
    label = Variable(torch.ones((2,)).long().cuda())
    print(loss(a, label))