import numpy as np
import pandas as pd
import scvi
from anndata import AnnData
import random


class metrics:
    def __init__(self, adata):
        self.adata = adata
    
    def silhouette_labels(
    self,
    labels_key: str = "labels_bench",
    metric: str = "euclidean",
    embedding: str = "X_scvi",
    scale: bool = True,
    ) -> float:
        """
        Wrapper of Silhoutte score from sklearn, with respect to observed cell type labels.
        The score is scaled into the range [0, 1], where larger values indicate better, denser clusters.
        """
        from sklearn.metrics import silhouette_score

        if embedding not in self.adata.obsm.keys():
            raise KeyError(f"{embedding} not in obsm")
        asw = silhouette_score(
            self.adata.obsm[embedding], self.adata.obs[labels_key], metric=metric
        )
        if scale:
            asw += 1
            asw /= 2
        return asw

    def silhouette_batch(
        self,
        batch_key: str = "batch_bench",
        labels_key: str = "labels_bench",
        metric: str = "euclidean",
        embedding: str = "X_scvi",
        scale: bool = True,
    ):
        """
        Silhouette score of batch labels subsetted for each group.
        batch_ASW = 1 - abs(ASW), so score of 1 means ideally mixed, 0, not mixed.
        Parameters
        ----------
        adata
            AnnData object
        batch_key
            Key in `adata.obs` containing batch information
        labels_key
            Key in `adata.obs` containing cell type labels
        metric
            A valid sklearn metric
        embedding
            Key in `adata.obsm` containing scvi-tools generated low-dim representation
        Returns
        -------
        all_scores
            Absolute silhouette scores per group label
        group_means
            Mean per cell type (group)
        """
        from sklearn.metrics import silhouette_score

        if embedding not in self.adata.obsm.keys():
            raise KeyError(f"{embedding} not in obsm")

        groups = self.adata.obs[labels_key].unique()
        sil_all = pd.DataFrame(columns=["silhouette_score"], index=groups)

        for group in groups:
            adata_group = self.adata[self.adata.obs[labels_key] == group]
            if adata_group.obs[batch_key].nunique() == 1:
                continue
            sil_per_group = silhouette_score(
                adata_group.obsm[embedding], adata_group.obs[batch_key], metric=metric
            )
            if scale:
                sil_per_group = (sil_per_group + 1) / 2
                sil_per_group = np.abs(sil_per_group)
                sil_per_group = 1 - sil_per_group

            sil_all.loc[group, "silhouette_score"] = sil_per_group
        return sil_all

    def lisi(
        self,
        batch_key: str = "batch_bench",
        labels_key: str = "labels_bench",
        embedding: str = "X_scvi",
        n_jobs: int = 1,
    ) -> float:
        """
        Wrapper of compute_lisi from harmonypy package. Higher is better.
        Suppose one of the columns in metadata is a categorical variable with 3 categories.
            - If LISI is approximately equal to 3 for an item in the data matrix,
            that means that the item is surrounded by neighbors from all 3
            categories.
            - If LISI is approximately equal to 1, then the item is surrounded by
            neighbors from 1 category.
        Returns
        -------
        iLISI
            lisi computed w.r.t. batches
        cLISI
            lisi computed w.r.t. labels
        """
        from harmonypy import compute_lisi
        from sklearn.utils import parallel_backend

        if embedding not in self.adata.obsm.keys():
            raise KeyError(f"{embedding} not in obsm")

        if n_jobs != 1:
            with parallel_backend("threading", n_jobs=n_jobs):
                lisi = compute_lisi(
                    self.adata.obsm[embedding],
                    metadata=self.adata.obs,
                    label_colnames=[batch_key, labels_key],
                )
        else:
            lisi = compute_lisi(
                self.adata.obsm[embedding],
                metadata=self.adata.obs,
                label_colnames=[batch_key, labels_key],
            )
        return lisi

    def compute_ari(self, labels_key: str = "labels_bench") -> float:
        """
        Adjusted rand index computed at various clustering resolutions, max reported.
        Parameters
        ----------
        adata
            AnnData object
        labels_key
            Key in `adata.obs` containing cell type labels
        """
        from sklearn.metrics.cluster import adjusted_rand_score

        group1 = self.adata.obs[labels_key]
        aris = []
        keys = self.adata.obs.keys()
        for k in keys:
            group2 = self.adata.obs[k]
            ari = adjusted_rand_score(group1, group2)
            aris.append(ari)

        return np.max(aris)

    def compute_nmi(self, labels_key: str = "labels_bench") -> float:
        """
        Adjusted rand index computed at various clustering resolutions, max reported.
        Parameters
        ----------
        adata
            AnnData object
        labels_key
            Key in `adata.obs` containing cell type labels
        """
        from sklearn.metrics.cluster import normalized_mutual_info_score

        group1 = self.adata.obs[labels_key]
        nmis = []
        keys = self.adata.obs.keys()
        for k in keys:
            group2 = self.adata.obs[k]
            nmi = normalized_mutual_info_score(group1, group2)
            nmis.append(nmi)

        return np.max(nmis)
    
    def ranking(self, model, donors, donor_key, label_key):
        """
        Helper function for get_batchy_genes. Computes the s2b gene rankings for one trial of k donors.
        Parameters
        ----------
        model
            Model used to train adata.
        donors
            List of donors in subset.
        donor_key
            Key in `adata.obs` containing donor labels
        label_key
            Key in `adata.obs` containing cell type labels
        """
        subset = self.adata[self.adata.obs[donor_key].isin(donors)]
        subset = subset.copy()
        scvi.model.SCVI.setup_anndata(
            subset,batch_key=donor_key,labels_key=label_key
        )
        genes = self.adata.var.index

        df = pd.DataFrame(columns = range(len(donors)))
        for i,donor in enumerate(donors):
            df[i] = model.get_normalized_expression(subset,transform_batch=donor).median()
        
        s2b = df.std(axis=1).divide(df.mean(axis=1))
        s2b = s2b.sort_values(ascending=False)
        s2b_ranks = pd.DataFrame({'s2b':s2b,'rank':range(len(genes))})

        ranks = []
        for gene in genes:
            ranks += [s2b_ranks.loc[gene]['rank']]
        return ranks

    def get_batchy_genes(
        self, 
        model, 
        donor_key: str = 'patient_id', 
        label_key: str = 'initial_clustering',
        n_donors_per_trial: int = 10, 
        n_trials: int = 10):
        """
        Returns dataframe of genes ranked by sensitivity to batch, along with their mean rank and std.
        Parameters
        ----------
        model
            Model used to train adata.
        biological_metadata
            etc
        donor_key
            Key in `adata.obs` containing donor labels
        label_key
            Key in `adata.obs` containing cell type labels
        n_donors_per_trial
            Number of donors in one trial.
        n_trials
            Number of trials.
        """
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        
        donors = self.adata.obs[donor_key].unique().tolist()
        genes = self.adata.var.index
        consensus_ranks = np.empty((len(genes),n_donors_per_trial))
        for i in range(n_trials):
            sample = random.sample(donors,n_donors_per_trial)
            ranks = self.ranking(model,sample,donor_key,label_key)
            consensus_ranks[:,i] = ranks
        
        mean_std = []
        for g in range(len(genes)):
            mean_std += [[consensus_ranks[g,:].mean(),consensus_ranks[g,:].std()]]
        mean_std = np.array(mean_std)
        df = pd.DataFrame(data={'gene':genes.to_list(),'mean_rank':mean_std[:,0],'mean_std':mean_std[:,1]})
        return df.sort_values('mean_rank')