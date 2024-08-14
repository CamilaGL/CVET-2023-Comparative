# CVET-2023-Comparative

This repository contains the code to perform bAVM nidus extraction as reported in [Comparative Study of Automated Algorithms for Brain Arteriovenous Malformation Nidus Extent Identification Using 3DRA](https://link.springer.com/article/10.1007/s13239-023-00688-w).

Starting from a binary nifti file with a brain vessel segmentation, it is possible to identify the bAVM nidus through a morphological approach or a skeleton-based approach.


# Conda environments

## vmtknetwork
Required to use the vmtknetworkextraction script.

### Created with:
    conda install python=3.6 scikit-image simpleitk vmtk -c simpleitk -c vmtk
    conda install llvm=3.3
    pip install -e . #to install project 

## nidusextraction
To perform the rest of operations.

### Created with:
    conda install edt -c conda-forge python=3.9
    conda install simpleitk -c simpleitk
    conda install vtk
    conda install vmtk -c conda-forge
    conda install scikit-image -c conda-forge
    pip install -e . #to install project

# Usage

## Skeletonisation

### Thinning
(Activate the nidusextraction environment)

`cvet_thinning -ifile "(path to nifti file with segmentation)" -ofile "(path to save .vtp file)"`

### vmtknetworkextraction
(Activate the vmtknetwork environment to skeletonise with the vmtknetworkextraction script)

`cvet_vmtknetwork -ifile "(path to nifti file with segmentation)" -ofile "(path to save .vtp file)"`


## Nidus extraction
(Activate the nidusextraction environment)

### Morphology-based
To perform nidus extraction with the morphological method (CHENOUNE 2019):<br>
`cvet_morph -ifile "(path to nifti file with segmentation)" -ofile "(path to save nifti file)" [-rad kernel_radius]`

### Skeleton-based
To perform nidus extraction with the skeleton-based method (BABIN 2018). **Make sure to perform skeletonisation beforehand**:<br>
`cvet_skeextractor -ifile "(path to nifti file with segmentation)" -iske "(path to .vtp file with skeletonisation)" -ofile "(path to save nifti file with nidus) -method "vmtk"/"skeleton" -spider "boundingbox"/"hull"/"spheres" [-next discard_firs_n_spiders]`


# Citation
If you find our paper useful, please use the following BibTeX entry for citation:

## Authors
**Camila García**, Ana Paula Narata, Jianmin Liu, Yibin Fang, Ignacio Larrabide

### Bibtex
@article{garcia2023comparative,
  title={Comparative Study of Automated Algorithms for Brain Arteriovenous Malformation Nidus Extent Identification Using 3DRA},
  author={Garc{\'\i}a, Camila and Narata, Ana Paula and Liu, Jianmin and Fang, Yibin and Larrabide, Ignacio},
  journal={Cardiovascular Engineering and Technology},
  volume={14},
  number={6},
  pages={801--809},
  year={2023},
  publisher={Springer}
}