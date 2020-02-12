from itertools import combinations

import numpy as np
import torch


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(
        1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)
    return distance_matrix


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """
    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(
                len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """
    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances,
                                        len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """
    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """
    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind]
                             for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


def set_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """
    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])),
                    torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=hardest_negative,
                                           cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=random_hard_negative,
                                           cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(
        margin=margin, negative_selection_fn=lambda x: semihard_negative(x, margin), cpu=cpu)


import matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as STSNE
from openTSNE import TSNE as OTSNE
from openTSNE.callbacks import ErrorLogger

import matplotlib.pyplot as plt

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf'
]


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    for i in range(10):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)

def plot_embeddings_v2(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    for i in range(10):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.1, color=colors[i], marker='*')
        plt.scatter(embeddings[inds + 10000, 0], embeddings[inds + 10000, 1], alpha=0.7, color=colors[i], marker='+')
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)


def extract_embeddingsx(dataloader, model, cuda=True):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 32))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k + len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k + len(images)] = target.numpy()
            k += len(images)
#         embeddings = reduce_dimension(embeddings, reduction)
    return embeddings, labels


def reduce_dimension(embeddings, reduction='pca'):
    if reduction == 'pca':
        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(embeddings)
    elif reduction == 'tsne':
        otsne = OTSNE(initialization='pca',
                      n_jobs=8,
                      callbacks=ErrorLogger(),
                      negative_gradient_method='bh')
        embeddings = otsne.fit(embeddings)
#         stsne = STSNE()
#         embeddings = stsne.fit_transform(embeddings)
    elif reduction == 'none':
        pass
    else:
        raise Exception
    return embeddings
        
def show_statistics(data_loader, model):
    from luna import Aggregator

    val_embeddings, val_labels = extract_embeddingsx(data_loader, model, 'none')
    val_embeddings = torch.from_numpy(val_embeddings).cuda()
    val_labels = torch.from_numpy(val_labels).cuda()

    ctr_embeddings = torch.zeros(10, 32).cuda()

    norms = torch.sum(val_embeddings ** 2, dim=1) ** 0.5
    print("all median: {:.2f}".format(
        torch.median(norms).item()
    ))

    inner_agg = Aggregator()
    for label_id in range(10):
        label_embeddings = val_embeddings[val_labels == label_id]
        num = label_embeddings.size(0)
        ctr = label_embeddings.mean(0)
        ctr_embeddings[label_id] = ctr
        dists = torch.sum((label_embeddings - ctr) ** 2, dim=1) ** 0.5
    #     print(torch.mean(dists).item(), torch.std(dists).item())
        sort_dists = dists.sort()[0]
        inner_agg.aggregate(("median", torch.median(dists).item()),
                            ("01", sort_dists[int(num * 0.01)].item()),
                            ("10", sort_dists[int(num * 0.1)].item()),
                            ("90", sort_dists[int(num * 0.9)].item()),
                            ("99", sort_dists[int(num * 0.99)].item()),
                            ("std",  torch.std(dists).item()))

    intra_agg = Aggregator()
    for label_id in range(10):
        dists = torch.sum((ctr_embeddings - ctr_embeddings[label_id]) ** 2, dim=1) ** 0.5
        sort_dists = dists.sort()[0]
        intra_agg.aggregate(("median", torch.median(dists).item()),
                            ("1", sort_dists[1].item()),
                            ("2", sort_dists[2].item()),
                            ("3", sort_dists[3].item()),
                            ("std", dists.std().item())
                        )
    # print(intra_dist)
    print("inner 1%: {:.2f}, 10%: {:.2f}, 90%: {:.2f}, 99%: {:.2f}, std: {:.2f}".
        format(inner_agg.mean("01"), 
                inner_agg.mean("10"), 
                inner_agg.mean("90"),
                inner_agg.mean("99"),
                inner_agg.mean("std")))
    print("intra 1: {:.2f}, 2: {:.2f}, 3: {:.2f}, std: {:.2f}".
        format(intra_agg.mean("1"), 
                intra_agg.mean("2"),
                intra_agg.mean("3"),
                intra_agg.mean("std")))
    print("inner: {:.2f}, intra: {:.2f}, intra/inner: {:.2f}".
        format(inner_agg.mean('median'), intra_agg.mean('median'), 
                intra_agg.mean('median')/inner_agg.mean('median')))