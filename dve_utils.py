import tps
import torch.nn.functional as F
import torch

class LossWrapper(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def __call__(self, *a, **kw):
        return self.fn(*a, **kw)

def dense_correlation_loss_dve(feats1,feats2, meta, pow=0.5, fold_corr=False, normalize_vectors=True):

    device = feats1.device

    # Grid (B,H,W,2): For each pixel in im1, where did it come from in im2
    if 'grid' in meta:
        grid = meta['grid'].to(device)
    else:
        grid = meta.to(device)

    H_input = grid.shape[1]
    W_input = grid.shape[2]


    B, C, H, W = feats1.shape
    h, w = H, W

    stride = H_input // H

    xxyy = tps.spatial_grid_unnormalized(H_input, W_input).to(device)
    batch_grid_u = tps.grid_unnormalize(grid, H_input, W_input)
    batch_grid_u = batch_grid_u[:, ::stride, ::stride, :]


    # if fold_corr:
    #     """This function computes the gradient explicitly to avoid the memory
    #     issues with using autorgrad in a for loop."""
    #     from model.folded_correlation_dve import DenseCorrDve
    #     dense_corr = DenseCorrDve.apply
    #     return dense_corr(feats1, feats2, xxyy, batch_grid_u, stride,
    #                       normalize_vectors, pow)

    loss = 0.
    for b in range(B):
        f1 = feats1[b].reshape(C, H * W)  # source
        f2 = feats2[b].reshape(C, h * w)  # target
        fa = feats1[(b + 1) % B].reshape(C, h * w)  # auxiliary

        if normalize_vectors:
            f1 = F.normalize(f1, p=2, dim=0) * 20
            f2 = F.normalize(f2, p=2, dim=0) * 20
            fa = F.normalize(fa, p=2, dim=0) * 20

        corr = torch.matmul(f1.t(), fa)
        corr = corr.reshape(H, W, h, w)
        smcorr = F.softmax(corr.reshape(H, W, -1), dim=2).reshape(corr.shape)
        smcorr_fa = smcorr[None, ...] * fa.reshape(-1, 1, 1, h, w)
        del smcorr

        f1_via_fa = smcorr_fa.sum((3, 4)).reshape(C, H * W)
        del smcorr_fa

        corr2 = torch.matmul(f1_via_fa.t(), f2).reshape(corr.shape)
        smcorr2 = F.softmax(corr2.reshape(H, W, -1), dim=2).reshape(corr.shape)
        del corr2

        with torch.no_grad():
            diff = batch_grid_u[b, :, :, None, None, :] - \
                    xxyy[None, None, ::stride, ::stride, :]
            diff = (diff * diff).sum(4).sqrt()
            diff = diff.pow(pow)

        L = diff * smcorr2

        loss += L.float().sum()
        torch.cuda.empty_cache()

    return loss / (H * W * B)

