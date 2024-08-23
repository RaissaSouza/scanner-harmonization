# HarmonyTM: Multi-center data harmonization applied to distributed learning for Parkinson’s disease classification
<div align="center">

</div>

<p align="center">
<img src="harmonyTM.png?raw=true">
</p>


Implementation of a data harmonization method for distributed learning using the Travelling Model for unlearning scanner information while training a Parkinson's disease classifier that is under review at the Journal of Medical Imaging: "[HarmonyTM: Multi-center data harmonization applied to distributed learning for Parkinson’s disease classification] (link soon).

Our code here is based on the investigation of a Parkinson's disease classification using non-identical distribution across 83 centers that used 23 different scanners to acquire the data.

If you find our framework, code, or paper useful to your research, please cite us!
```
@article{
}

```
```

```

### Abstract 
Abstract  
Purpose
Distributed learning is widely used to comply with data-sharing regulations and access diverse datasets for training machine learning (ML) models. The travelling model (TM) is a distributed learning approach that sequentially trains with data from one center at a time, which is especially advantageous when dealing with limited local datasets. However, a critical concern emerges when centers utilize different scanners for data acquisition, which could potentially lead models to exploit these differences as shortcuts. While data harmonization can mitigate this issue, current methods typically rely on large or paired datasets, which can be impractical to obtain in distributed setups.  
Approach  
In this work, we introduced HarmonyTM, a data harmonization method tailored for the travelling model (TM). HarmonyTM effectively mitigates bias in the model's feature representation while retaining crucial disease-related information, all without requiring extensive datasets. Specifically, we employed adversarial training to "unlearn" bias from the features used in the model for classifying Parkinson's disease (PD). We evaluated HarmonyTM using multi-center three-dimensional neuroimaging datasets from 83 centers using 23 different scanners. 
Results  
Our results show that HarmonyTM improved PD classification accuracy from 72% to 76% and reduced (unwanted) scanner classification accuracy from 53% to 30% in the TM setup.  
Conclusion  
HarmonyTM is a method tailored for harmonizing three-dimensional neuroimaging data within the TM approach, aiming to minimize shortcut learning in distributed setups. This prevents the disease classifier from leveraging scanner-specific details to classify patients with or without PD — a key aspect for deploying ML models for clinical applications. 

### Method details
We use the state-of-the-art simple fully convolutional network (SFCN) (doi: 10.1016/J.MEDIA.2020.101871) as our deep learning architecture. The Adam optimizer with an initial learning rate of 0.001, an exponential decay after every cycle, and batch size 5 or less when fewer than 5 datasets were available at the center was used during training.

Before removing domain-specific details, it is necessary to pre-train the network components with the travelling model approach. Therefore, the encoder and disease classification head are initially trained until convergence (code in folder **encoder_pd**). Following this, the encoder is frozen, and the scanner classification head is trained until convergence (code in folder **scanner**). Finally, utilizing these pre-trained models, the scanner harmonization procedure is implemented in three steps as follows (code in folder **unlearning**). It is important to highlight that these steps are performed for each batch, such that the three training steps occur at each center before transferring the model to the next center (see workflow figure). 

1. Optimize the encoder and disease classification head for the PD classification task. 

2. Optimize the scanner classification head for identifying scanners from the feature representation of the frozen encoder that is trained in step 1. 

3. Optimize the encoder by employing an adversarial confusion loss to eliminate scanner-specific information. This loss guides the scanner classification head output toward chance-level performance. In essence, chance-level means that the model would make predictions purely by random guessing, eliminating any shortcuts related to scanners that might be exploited for disease classification. 

## Training code 

Every folder has a datagenerator.py file, which loads MRI data from a CSV file with their paths and other relevant information, such as PD group, center, and scanner type.

The code used to pre-train the PD classifier is in: 
```bash
├── code/encoder_pd
│   ├── enc_PD_train.py # for centralized approach
│   ├── enc_PD_train_distributed.py # for travelling model approach
```

The code used to pre-train the scanner classifier is in: 
```bash
├── code/scanner
│   ├── SC_train.py # for centralized approach
│   ├── SC_train_distributed.py # for travelling model approach
```

The code used for data harmonization, i.e., unlearn scanner while retaining PD, is in: 
```bash
├── code/unlearn
│   ├── main_central.py # for centralized approach
│   ├── main_distributed.py # for travelling model approach
```

## Evaluation
The code used for evaluation is in: 
```bash
├── code/inference
│   ├── inference_pd.py #for single model PD inference can be used for a centralized approach or a specific model in the travelling model approach
│   ├── inference_pd_distributed.py #iterate through all models saved (one per cycle) during travelling model training for PD inference
│   ├── inference_sc.py #for single model scanner inference can be used for a centralized approach or a specific model in the travelling model 
│   ├── inference_sc_distributed.py #iterate through all models saved (one per cycle) during travelling model training for scanner inference
```
## Important
Every .py file has its own arguments, such as path for training_set, test_set, number of cycles, number of epochs, batch_size, etc. Therefore, to run each file, just change its arguments and run as the following example:

```
python end_PD_train_pd.py -fn_train ./training_set.csv -fn_test ./testing_set.csv -batch_size 5
```

## Environment 
Our code for the Keras model pipeline used: 
* Python 3.10.6
* pandas 1.5.0
* numpy 1.23.3
* scikit-learn 1.1.2
* simpleitk 2.1.1.1
* tensorflow-gpu 2.10.0
* cudnn 8.4.1.50
* cudatoolkit 11.7.0

GPU: NVIDIA GeForce RTX 3090


## Resources
* Questions? Open an issue or send an [email](mailto:raissa_souzadeandrad@ucalgary.ca?subject=PD-travelling-model).
