from omegaconf import DictConfig
from hydra.utils import instantiate
import pytorch_lightning as pl
import hydra
import torch
import torch.nn.functional as F
from sklearn.cluster import (
    KMeans, SpectralClustering, spectral_clustering, AgglomerativeClustering
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
import numpy as np
from scipy.special import softmax

from ssc import SparseSubspaceClusteringOMP
from ssc import dim_reduction
from kymatio import Scattering2D
from torchvision.transforms import Resize

from mixmate.mixture import MixMATEv2
from mixmate.mixture.modulev2 import SingleAutoencoder


@hydra.main(config_path='configs/', config_name='trainv2')
def trainv2(config: DictConfig) -> None:
    dataset = instantiate(config.dataset)
    mixmate = MixMATEv2(**config.mixmate)
    trainer = pl.Trainer(**config.trainer)

    imgs_lst, labs_lst = zip(*[batch for batch in dataset.train_loader])
    imgs = torch.cat(imgs_lst, dim=0).numpy()
    true_labs = torch.cat(labs_lst, dim=0).numpy()

    ### INITIALIZATION ###
    # Only get clean images for initialization (if some are missing data)
    mask = np.all(imgs.reshape(len(imgs), -1) >= 0.0, axis=-1)
    imgs = imgs[mask]
    true_labs = true_labs[mask]

    # Center data points
    imgs_proc = torch.tensor(imgs, dtype=torch.float)
    dim = int(np.sqrt(mixmate.input_size))
    imgs_proc = imgs_proc.view(-1, 1, dim, dim)
    imgs_proc = imgs_proc - torch.mean(imgs_proc, dim=(1, 2, 3), keepdim=True)
    imgs = imgs.reshape(-1, mixmate.input_size)

    subset_size = config.init_subset_size
    idx = np.random.choice(np.arange(len(imgs_proc)), replace=False, size=(subset_size,))
    
    # Apply initialization algorithm
    if config.init_alg == 'kmeans':
        clusterer = KMeans(n_clusters=mixmate.num_components)
        labels = clusterer.fit_predict(imgs[idx])

    elif config.init_alg == 'spectral':
        print("making graph...")
        graph = kneighbors_graph(imgs[idx], 1000)
        graph = (graph + graph.T) / 2
        print("running spectral clustering...")
        labels = spectral_clustering(graph, n_clusters=mixmate.num_components)
    
    elif config.init_alg == 'ssc':
        scattering = Scattering2D(J=3, shape=(dim, dim))
        scattered = scattering(imgs_proc[idx])
    
        # scattering transform normalization
        scattered = scattered.numpy().reshape(subset_size, scattered.shape[2], -1)
        image_norm = np.linalg.norm(scattered, ord=np.inf, axis=2, keepdims=True)  # infinity norm of each transform
        scattered = scattered / image_norm  # normalize each scattering transform to the range [-1, 1]
        scattered = scattered.reshape(subset_size, -1)  # fatten and concatenate all transforms

        # dimension reduction
        scattered = dim_reduction(scattered, 500)  # dimension reduction by PCA

        model = SparseSubspaceClusteringOMP(n_clusters=10, affinity='symmetrize', n_nonzero=5, thr=1.0e-5)

        labels = model.fit_predict(scattered)
        print(np.bincount(labels))

    # Compute subset clustering accuracy
    print('Cluster Acc', mixmate._compute_cluster_accuracy(labels, true_labs[idx], 10))
    
    # Initialize dictionaries with subset clusters
    for i in range(mixmate.num_components):
        imgs_i = torch.tensor(imgs[idx][labels == i], dtype=torch.float)
        ordering = torch.randperm(imgs_i.size(dim=0))
        imgs_i = imgs_i[ordering][:mixmate.hidden_size]
        imgs_i = F.normalize(imgs_i - imgs_i.mean(dim=1, keepdim=True), dim=1)
        mixmate.W.data[i] = imgs_i.T

    # Make column norms equal to 1
    mixmate._normalize()

    # Log initial performance 
    trainer.validate(mixmate, dataset.valid_loader)

    # Train the model
    trainer.fit(mixmate, dataset.train_loader, dataset.valid_loader)

if __name__ == "__main__":
    trainv2()
