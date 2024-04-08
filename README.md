# Alzheimer’s disease detection in PSG signals

This repository contains the implementations of semi-supervised SMATE and TapNet models, supervised XCM network 
as described in the papers [SMATE: Semi-Supervised Spatio-Temporal Representation Learning on Multivariate Time Series](https://www.jingweizuo.com/publication/SMATE_ICDM2021.pdf)), [TapNet: Multivariate Time Series Classification with Attentional Prototype Network](https://ojs.aaai.org/index.php/AAAI/article/view/6165), [XCM: An Explainable Convolutional Neural Network for Multivariate Time Series Classification](https://hal.inria.fr/hal-03469487/document), and an implementation of the Hidden Markov Models for an unsupervised temporal series classification. 

## Abstract 
Alzheimer’s disease (AD) and sleep disorders exhibit a close association, where disruptions in sleep patterns often precede the onset of Mild Cognitive Impairment (MCI) and early-stage AD. This study delves into the potential of utilizing sleep-related electroencephalography (EEG) signals acquired through polysomnography (PSG) for the early detection of AD. Our primary focus is on exploring semi-supervised Deep Learning techniques for the classification of EEG signals due to the clinical scenario characterized by the limited data availability. The methodology entails testing and comparing the performance of semi-supervised SMATE and TapNet models, benchmarked against the supervised XCM model, and unsupervised Hidden Markov Models (HMMs). The study highlights the significance of spatial and temporal analysis capabilities, conducting independent analyses of each sleep stage. 

![modelos_png](https://github.com/LorenaGallego/DL4ADpred/assets/149390061/179873bd-202e-4643-a3c8-f49d161fc968)


## Requirements
Networks of the four models have been implemented in Python 3.8 with the following packages:
* keras = 2.2.4
* tensorflow = 1.14.0 with CUDA 10.2
* pytorch = 2.0.1 with CUDA 11.7
* pyhhmm
* matplotlib
* numpy
* pandas
* pyyaml
* scikit-learn
* seaborn
* graphviz
* scikit-learn




## Data 
The data used concatenate 4 different PSG databases, 1 for patients with AD and 3 public databases to obtain studies of healthy patients where only 4 EEG channels were taken. These databases were preprocessed to avoid biases in the experiments, and were separated into the different sleep stages (N1, N2, N3, REM) to perform independent tests.
Databases of healthy patients can be found in: 
* [ISRUC-SLEEP Dataset (Subgroup-III)](https://doi.org/10.1016/j.cmpb.2015.10.013)
* [Dream Open Dataset-Healthy (DOD-H)](https://doi.org/10.1109/TNSRE.2020.3011181)
* [Sleep Disorders Research Center (SDRC) Dataset](https://doi.org/10.17632/3hx58k232n.4)


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
python main.py
```

**HMM model**
```
python HMM_att_train.py
```


## Citation
```
@misc{gallegoviñarás2024alzheimers,
      title={Alzheimer's disease detection in PSG signals}, 
      author={Lorena Gallego-Viñarás and Juan Miguel Mira-Tomás and Anna Michela-Gaeta and Gerard Pinol-Ripoll and Ferrán Barbé and Pablo M. Olmos and Arrate Muñoz-Barrutia},
      year={2024},
      eprint={2404.03549},
      archivePrefix={arXiv},
      primaryClass={eess.SP}
}
```
