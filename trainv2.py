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

from ssc.cluster.selfrepresentation import SparseSubspaceClusteringOMP, ElasticNetSubspaceClustering
from ssc.decomposition.dim_reduction import dim_reduction
from kymatio import Scattering2D
from torchvision.transforms import Resize

from mixmate.mixture import MixMATEv2
from mixmate.mixture.modulev2 import SingleAutoencoder


@hydra.main(config_path='configs/', config_name='trainv2')
def trainv2(config: DictConfig) -> None:
    dataset = instantiate(config.dataset)
    mixmate = MixMATEv2(**config.mixmate)
    trainer = pl.Trainer(**config.trainer)

    num_prelabeled = config.num_prelabeled
    if num_prelabeled is not None:
        imgs_lst, labs_lst = zip(*[batch for batch in dataset.valid_loader])
        imgs_all = torch.cat(imgs_lst, dim=0).numpy()
        true_labs = torch.cat(labs_lst, dim=0).numpy()
        for i in range(mixmate.num_components):
            imgs = torch.tensor(
                imgs_all[true_labs == i],
                dtype=torch.float
            )

            if config.use_svd:
                V = imgs.view((-1, config.mixmate.input_size))[:num_prelabeled].svd().V
                mixmate.W.data[i] = V[:, :mixmate.hidden_size]
            else:
                ordering = torch.randperm(imgs.size(dim=0))
                imgs = imgs[ordering]
                imgs = imgs.view((-1, config.mixmate.input_size))[:num_prelabeled][:mixmate.hidden_size]
                imgs = F.normalize(imgs - imgs.mean(dim=1, keepdim=True), dim=1)
                mixmate.W.data[i] = imgs.T
        
    else:
        # From random data points
        # imgs = torch.tensor(
        #         dataset._dataset.data,
        #         dtype=torch.float
        # )
        # for i in range(mixmate.num_components):
        #     idx = torch.randperm(imgs.size(0))[:mixmate.hidden_size]
        #     imgs_i = imgs.view((-1, mixmate.input_size))[idx]
        #     imgs_i = F.normalize(imgs_i - imgs_i.mean(dim=1, keepdim=True), dim=1)
        #     mixmate.W.data[i] = imgs_i.T

        # From K-means pretrained auto-encoder
        # mixmate_pretrain = SingleAutoencoder(
        #     num_layers=15, 
        #     input_size=784, 
        #     hidden_size=160,
        #     step_size=config.mixmate.step_size,
        #     prox={
        #         '_target_': 'mixmate.ista.GroupSparseBiasedReLU', 
        #         'num_groups': 10, 
        #         'group_size': 16, 
        #         'group_sparse_penalty': 5
        #     },
        #     # prox={
        #     #     '_target_': 'mixmate.ista.DoubleSidedBiasedReLU', 
        #     #     'sparse_penalty': 0.75
        #     # },
        #     beta=config.mixmate.beta,
        #     lr=config.mixmate.lr
        # )
        # imgs = torch.tensor(
        #         dataset._dataset.data,
        #         dtype=torch.float
        # )

        # train_network = True
        # if train_network:
        #     # Sparse init
        #     # V = imgs.view(imgs.shape[0], -1).svd().V
        #     # mixmate_pretrain.W.data = V[:, :mixmate_pretrain.hidden_size].unsqueeze(dim=0)

        #     # Group sparse init
        #     hidden_size = mixmate_pretrain.hidden_size
            
        #     group_size = mixmate_pretrain.prox.group_size
        #     num_groups = mixmate_pretrain.prox.num_groups

        #     V = imgs.view(imgs.shape[0], -1).svd().V
        #     P = torch.empty(num_groups, group_size, group_size)
        #     torch.nn.init.orthogonal_(P)
        #     mixmate_pretrain.W.data = (P.view(hidden_size, group_size) @ V[:, :group_size].T).T.unsqueeze(dim=0)
        
        #     trainer_pretrain = pl.Trainer(**config.trainer)
        #     trainer_pretrain.fit(mixmate_pretrain, dataset.train_loader, dataset.valid_loader)
        #     torch.save(mixmate_pretrain.W.data, 'weights.pt')
        # else:
        #     mixmate_pretrain.W.data = torch.load('/home/ec2-user/mixmate/weights.pt')
        
        # codes = []
        # datas = []
        # true_labs = []
        # for data, true_lab in dataset.train_loader:
        #     _, _, code, _, _ = mixmate_pretrain(data)
        #     true_labs.append(true_lab.numpy())
        #     datas.append(data.numpy())
        #     codes.append(code.detach().numpy()[:, 0, :])
        # datas = np.concatenate(datas, axis=0)
        # codes = np.concatenate(codes, axis=0)

        # Normalize codes
        # codes = np.abs(codes)
        # codes = codes / codes.sum(axis=1).reshape(-1, 1)
        # codes = np.nan_to_num(codes)

        # true_labs = np.concatenate(true_labs, axis=0)
        # kmeans = KMeans(n_clusters=mixmate.num_components)
        # labels = kmeans.fit_predict(codes)
        # print('Kmeans Acc', mixmate._compute_cluster_accuracy(labels, true_labs, 10))

        # for i in range(mixmate.num_components):
        #     datas_i = datas[labels == i]
        #     distances = kmeans.transform(codes[labels == i])
        #     ordering = torch.tensor(np.argsort(distances[:, i]), dtype=torch.long)
        #     # ordering = np.arange(datas_i.shape[0])
        #     # np.random.shuffle(ordering)
        #     imgs_i = torch.tensor(datas_i).view((-1, 28 * 28))[ordering][:mixmate.hidden_size]
        #     imgs_i = F.normalize(imgs_i - imgs_i.mean(dim=1, keepdim=True), dim=1)
        #     mixmate.W.data[i] = imgs_i.T

        imgs_lst, labs_lst = zip(*[batch for batch in dataset.valid_loader])
        imgs = torch.cat(imgs_lst, dim=0).numpy()
        true_labs = torch.cat(labs_lst, dim=0).numpy()

        # s = StandardScaler(with_std=False)
        # imgs_scaled = s.fit_transform(imgs)

        # resize imgs
        imgs_proc = torch.tensor(imgs, dtype=torch.float)
        # imgs_proc = Resize((32, 32))(imgs_proc)
        dim = int(np.sqrt(mixmate.input_size))
        imgs_proc = imgs_proc.view(-1, 1, dim, dim)
        imgs_proc = imgs_proc - torch.mean(imgs_proc, dim=(1, 2, 3), keepdim=True)

        imgs = imgs.reshape(-1, mixmate.input_size)

        # K-Means
        # clusterer = KMeans(n_clusters=mixmate.num_components)
        # labels = clusterer.fit_predict(imgs)

        # Spectral clustering
        # imgs = imgs[:10000]
        # print("making graph...")
        # graph = kneighbors_graph(imgs, 1000)
        # graph = (graph + graph.T) / 2
        # print("running spectral clustering...")
        # labels = spectral_clustering(graph, n_clusters=mixmate.num_components)

        # true_labs = np.array(dataset._dataset.targets)[:10000]

        # idx = np.random.choice(np.arange(60000), replace=False, size=(8000,))
        # view = TSNE(n_components=2).fit_transform(imgs_scaled[idx])
        # clusterer = KMeans(n_clusters=10)
        # clusterer = AgglomerativeClustering(n_clusters=10)
        # labels = clusterer.fit_predict(view)

        # SSC
        # subset_size = 4 * mixmate.num_components * mixmate.hidden_size
        subset_size = 2000
        idx = np.random.choice(np.arange(len(dataset._dataset)), replace=False, size=(subset_size,))
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
        # model = ElasticNetSubspaceClustering(
        #     n_clusters=10, affinity='nearest_neighbors', algorithm='spams', active_support=True, 
        #     gamma=200, tau=0.9
        # )

        labels = model.fit_predict(scattered)
        print(np.bincount(labels))

        print('Cluster Acc', mixmate._compute_cluster_accuracy(labels, true_labs[idx], 10))
        for i in range(mixmate.num_components):
            imgs_i = torch.tensor(imgs[idx][labels == i], dtype=torch.float)
            if config.use_svd:
                V = imgs_i.svd().V
                mixmate.W.data[i] = V[:, :mixmate.hidden_size]
            else:
                # distances = clusterer.transform(view[labels == i])
                # probs = softmax(-distances, axis=-1)
                # ordering = torch.tensor(np.argsort(distances[:, i]), dtype=torch.long)
                # ordering = torch.tensor(np.argsort(-probs[:, i]), dtype=torch.long)
                ordering = torch.randperm(imgs_i.size(dim=0))
                imgs_i = imgs_i[ordering][:mixmate.hidden_size]
                imgs_i = F.normalize(imgs_i - imgs_i.mean(dim=1, keepdim=True), dim=1)
                mixmate.W.data[i] = imgs_i.T

    mixmate._normalize()

    # data = dataset._dataset.data
    # num_data = len(data)
    # for i in range(mixmate.num_components):
    #     idx = np.random.choice(num_data, mixmate.hidden_size, replace=False)
    #     imgs = torch.tensor(data[idx], dtype=torch.float).detach().clone()
    #     mixmate.W.data[i] = F.normalize(imgs.view((-1, 28 * 28)).transpose(0, 1)

    trainer.validate(mixmate, dataset.valid_loader)
    trainer.fit(mixmate, dataset.train_loader, dataset.valid_loader)

if __name__ == "__main__":
    trainv2()
