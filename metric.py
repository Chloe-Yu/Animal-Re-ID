import torch
import numpy as np

def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
      normalized: True if passed in tensors are already normalized
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    x_norm, y_norm = x,y
      
    dist = 1- torch.mm(x_norm, y_norm.transpose(0,1))
    return dist

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

###################
#compute metric
def compute_mAP(index, good_index, remove_closest):
    ap = 0
    length = len(index)
    if remove_closest:
        length+=1
    cmc = torch.IntTensor(length).zero_()

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def evaluate_CMC(query_features, query_labels, gallery_features, gallery_labels, remove_closest = True,distance = 'euclidean',noramlized=True):
    CMC = torch.IntTensor(len(gallery_labels)).zero_()
    ap = 0.0
    if distance == 'euclidean':
        score = euclidean_dist(query_features,gallery_features)
    else:
        score = cosine_dist(query_features,gallery_features)
        
    for i in range(len(query_labels)):
        ap_tmp, CMC_tmp = evaluate_rerank(score[i,:], query_labels[i], gallery_labels,remove_closest)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC / len(query_labels)  # average CMC
    return CMC, ap, score


#######################################################################
# Evaluate one query after re-ranking
def evaluate_rerank(score, ql, gl,remove_closest):
    index = np.argsort(score)
    query_index = np.argwhere(gl == ql).flatten()
    good_index = query_index
    
    if remove_closest:
        id_ind = index[0]
        if torch.is_tensor(id_ind):
            id_ind = id_ind.item()
        index = index[1:] #discarded the one with shorted distance to not count itself
        
        good_index = np.delete(query_index, np.where(query_index == id_ind)[0])
    if len(good_index)!=0:
        ap,CMC_tmp = compute_mAP(index, good_index, remove_closest)
    else:
        return -1, [-1]
    return ap,CMC_tmp
