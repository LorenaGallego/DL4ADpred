# Alzheimer’s disease detection in PSG signals

This repository contains the implementations of semi-supervised SMATE and TapNet models, supervised XCM network 
as described in the papers [SMATE: Semi-Supervised Spatio-Temporal Representation Learning on Multivariate Time Series](https://www.jingweizuo.com/publication/SMATE_ICDM2021.pdf)), [TapNet: Multivariate Time Series Classification with Attentional Prototype Network](https://ojs.aaai.org/index.php/AAAI/article/view/6165), [XCM: An Explainable Convolutional Neural Network for Multivariate Time Series Classification](https://hal.inria.fr/hal-03469487/document), and an implementation of the Hidden Markov Models for an unsupervised temporal series classification. 

## Abstract 
Alzheimer’s disease (AD) and sleep disorders exhibit a close association, where disruptions in sleep patterns often precede the onset of Mild Cognitive Impairment (MCI) and early-stage AD. This study delves into the potential of utilizing sleep-related electroencephalography (EEG) signals acquired through polysomnography (PSG) for the early detection of AD. Our primary focus is on exploring semi-supervised Deep Learning techniques for the classification of EEG signals due to the clinical scenario characterized by the limited data availability. The methodology entails testing and comparing the performance of semi-supervised SMATE and TapNet models, benchmarked against the supervised XCM model, and unsupervised Hidden Markov Models (HMMs). The study highlights the significance of spatial and temporal analysis capabilities, conducting independent analyses of each sleep stage. 

## Data 


## Requirements
Networks of the four models have been implemented in Python 3.8 with the following packages:
* keras = 2.2.4
* matplotlib
* numpy
* pandas
* pyyaml
* scikit-learn
* seaborn
* tensorflow = 1.14.0 with CUDA 10.2
* graphviz
* scikit-learn
* pyhhmm
* pytorch = 2.0.1 with CUDA 11.7


## Usage
**SMATE model**
```
python SMATE_classifier_v2.py --ds_name DATASET_NAME
```

**TapNet model**
```
python train_3.py
```

**XCM model**

```
python main.py --config configuration/config.yml
```

**HMM model**

The current configuration file provides an example of classification with XCM on the Basic Motions UEA dataset 
with the configuration presented in the paper, and an example of a heatmap from Grad-CAM for the
first MTS of the test set. 

## Citation
```
@article{Fauvel21XCM,
  author = {Fauvel, K. and T. Lin and V. Masson and E. Fromont and A. Termier},
  title = {XCM: An Explainable Convolutional Neural Network for Multivariate Time Series Classification},
  journal = {Mathematics},
  year = {2021},
  volume = {9},
  number = {23}
}
```
