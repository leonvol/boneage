<p align="center"><img src="https://raw.githubusercontent.com/leonvol/bwki-boneage-experiments/main/imgs/logo.png" alt="BoneAge"></p>

This project is the world's first AI system to determine a person's age by analyzing 3D low-dose thorax CT images of the clavicle. It has higher accuracy and a wider age detection range than more traditional hand bone age assessment and is much faster than estimates of trained radiologists.

&rarr; Invitation to 2020's nationwide final, placed TOP 5

## Code structure overview
| module name                | function                                                                                          |
|----------------------------|---------------------------------------------------------------------------------------------------|
| batch_loader               | fast, parallelized loading, processing, augmenting and caching of CT images                       |
| train_framework            | framework to train and compare the performance of different net structures                        |
| vgg16_3d                   | implementation of a 3D VGG16 Net                                                                  |
| vgg16_attention_pretrained | pretrained 3D VGG16 Net with attention                                                            |
| alexnet_3d                 | implementation of a 3D Alexnet                                                                    |
| convert_crop               | automatically crop and convert DICOM data with segmentation point                                 |
| preprocessing              | helper functions for preprocessing                                                                |
| util                       | general helper functions                                                                          |
| clr_callback               | cyclic learning rate callback for keras                                                           |
| predict                    | prediction of not yet segmented CT images                                                         |

## Installation 
Installation of all needed dependencies by running
```bash
pip install -r requirements.txt
```

## Results
The best models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1ax7hesbNFF8awU1AiC-yM3L72oZdFj6l?usp=sharing)

|   | neural net structure                                       | learning rate     | Test-Set MAE in months |
|---|------------------------------------------------------------|-------------------|-------------------------|
| 1 | 3D VGG16, BN, 3 Dense*                                     | CLR [0.01, 0.001] | 23.14                   |
| 2 | 3D AlexNet, 4 Conv Layers, BN, 3 Dense                     | CLR [0.01, 0.001] | 23.76                   |
| 3 | 3D VGG16, BN, GlobalMaxPooling3D*                          | CLR [0.01, 0.001] | 25.60                   |
| 4 | VGG16 Attention**, ersten 3 Layer trainierbar, BN, 3 Dense | CLR [0.1, 0.01]   | 30.16                   |
| 5 | VGG16 Attention**, GlobalMaxPooling                        | CLR [0.1, 0.01]   | 32.43                   |
| ...                                                                                                          |

*modified, without pooling after the 4th block to allow for convolutions in the 5th block

**pretrained on RSNA Bone Age from [kaggle](https://www.kaggle.com/kmader/attention-on-pretrained-vgg16-for-bone-age)

## Acknowledgment
Thanks to LMU for the dataset
