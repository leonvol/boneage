# bwki-boneage-experiments
Experiments and submission code of project 'BoneAge' for competition BWKI

## Projekt
Das Projekt "BoneAge" löst das Problem der Alterserkennung einer Person anhand von CT-Aufnahmen des Schlüsselbeins.

## Übersicht
| Modul Name                 | Aufgabe                                                                                           |
|----------------------------|---------------------------------------------------------------------------------------------------|
| batch_loader               | Schnelles und parallelisiertes Laden, Preprocessen, Augmenten und Cachen von CT Bildern           |
| train_framework            | Simples trainieren und vergleichen verschiedener Netzstrukturen                                   |
| preprocessing              | Helper zum Preprocessen der 3d Daten                                                              |
| util                       | Allgemeine Helper                                                                                 |
| vgg16_3d                   | Implementation einer 3d VGG16 Netzstruktur, sowie Trainingscode                                   |
| vgg16_attention_pretrained | Integration eines vortrainierten VGG16 Netzwerkes, in ein 3d Modell, sowie Trainingscode          |
| convert_crop               | Code zum automatischen Konvertieren und Zuschneiden der DICOM Dateien                             |
| clr_callback               | Cyclic Learning Rate Keras Callback                                                               |
