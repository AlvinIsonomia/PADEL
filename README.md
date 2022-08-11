# PADEL
This paper is submitted to WSDM 2023, submission id is 

# Dataset
PADEL and baseline methods are implemented on the ``hpo_metab``, ``hpo_neuro``, and ``em_user`` datasets, firstly relsed by [SubGNN](https://www.dropbox.com/sh/zv7gw2bqzqev9yn/AACR9iR4Ok7f9x1fIAiVCdj3a?dl=0).
We provide these datasets in different data-efficient situations:
```
datasets-PADEL.7z
├─em_user
│      edge_list.txt
│      subgraphs.pth
│      subgraphs_10.pth
│      subgraphs_20.pth
│      subgraphs_30.pth
│      subgraphs_40.pth
│      subgraphs_50.pth
│
├─hpo_metab
│      edge_list.txt
│      subgraphs.pth
│      subgraphs_10.pth
│      subgraphs_20.pth
│      subgraphs_30.pth
│      subgraphs_40.pth
│      subgraphs_50.pth
│
└─hpo_neuro
        edge_list.txt
        subgraphs.pth
        subgraphs_10.pth
        subgraphs_20.pth
        subgraphs_30.pth
        subgraphs_40.pth
        subgraphs_50.pth
------ 
```

``edge_list.txt`` is the orignal edge list file for the base graph. ``subgraphs.pth`` is the original subgraph file with subgraph and labels.
``subgraphs_X0`` means the new subgraph file consisting of X0% of the training set and the original validation/ test set.


# Codes
We provide pseudo-code for random node diffusion and training process, and the source code will be released upon acceptance.
