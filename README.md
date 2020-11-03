<p align="center"><img src="https://raw.githubusercontent.com/leonvol/bwki-boneage-experiments/main/imgs/logo.png" alt="BoneAge"></p>

# bwki-boneage-experiments
Experiments and submission code of project 'BoneAge' for competition BWKI

## Projekt
Das Projekt "BoneAge" löst das Problem der Alterserkennung einer Person anhand von CT-Aufnahmen des Schlüsselbeins.

**Einladung zum Bundesfinale**

## Übersicht
| Modul Name                 | Aufgabe                                                                                           |
|----------------------------|---------------------------------------------------------------------------------------------------|
| batch_loader               | Schnelles und parallelisiertes Laden, Preprocessen, Augmenten und Cachen von CT Bildern           |
| train_framework            | Simples trainieren und vergleichen verschiedener Netzstrukturen                                   |
| preprocessing              | Helper zum Preprocessen der 3d Daten                                                              |
| util                       | Allgemeine Helper                                                                                 |
| vgg16_3d                   | Implementation einer 3d VGG16 Netzstruktur, sowie Trainingscode                                   |
| vgg16_attention_pretrained | Integration eines vortrainierten VGG16 Netzwerkes, in ein 3d Modell, sowie Trainingscode          |
| alexnet_3d                 | Implementation einer 3d Alexnet Netzstruktur, sowie Trainigscode                                  |
| convert_crop               | Code zum automatischen Konvertieren und Zuschneiden der DICOM Dateien                             |
| clr_callback               | Cyclic Learning Rate Keras Callback                                                               |
| predict                    | Einfache Prediction für nicht-segmentierte CT-Aufnahmen                                           |

## Installation 
Erstellung eines neuen pipenv und Installation der benötigten packages durch
```bash
pip install -r requirements.txt
```

## Prediction
Beispielscode zum Vorhersagen des Knochenalters einer CT-Aufnahme mit Segmentation, am Ende von `predict.py`
```python
from vgg16_3d import preprocessing, output_reshape # Importieren der gewünschten Netzstruktur und der zugehörigen Funktionen
ct = 'data/sample/' # Directory der dicom-Dateien
nrrd = 'data/sample.nrrd' # Pfad zur Segmentierungsdatei
weights = 'models/vgg16_3d/best' # Pfad zu den Dateien der Netzgewichte
dims = (130, 100, 15) # Dimensionen des Ausschnitts, sollte zur Verwendung mit dem vortrainierten Modellen (130, 100, 15) sein
predict(ct, nrrd, preprocessing, output_reshape, weights, dims)
```

## Ergebnisse
|   | Netzwerkstruktur                                           | Learning Rate     | Test-Set MAE in Monaten |
|---|------------------------------------------------------------|-------------------|-------------------------|
| 1 | 3D VGG16, BN, 3 Dense*                                     | CLR [0.01, 0.001] | 23.14                   |
| 2 | 3D AlexNet, 4 Conv Layers, BN, 3 Dense                     | CLR [0.01, 0.001] | 23.76                   |
| 3 | 3D VGG16, BN, GlobalMaxPooling3D*                          | CLR [0.01, 0.001] | 25.60                   |
| 4 | VGG16 Attention**, ersten 3 Layer trainierbar, BN, 3 Dense | CLR [0.1, 0.01]   | 30.16                   |
| 5 | VGG16 Attention**, GlobalMaxPooling                        | CLR [0.1, 0.01]   | 32.43                   |
| ...                                                                                                          |

*Modifiziert, ohne Pooling nach dem 4. Block um 5. Convolutional Block durchführen zu können

**Vortrainiert auf RSNA Bone Age, Quelle: https://www.kaggle.com/kmader/attention-on-pretrained-vgg16-for-bone-age

Die besten Modelle der einzelnen Netzstrukturen sind zu finden unter: https://drive.google.com/drive/folders/1ax7hesbNFF8awU1AiC-yM3L72oZdFj6l?usp=sharing

## Danksagung
LMU für den Datensatz
@pwesp für die grundlegende 3D Netzstruktur, welche wir abänderten
