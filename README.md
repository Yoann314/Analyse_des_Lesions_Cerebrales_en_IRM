# Titre du Projet: Analyse des Lésions Cérébrales en IRM

## Description
Ce projet vise à analyser les lésions cérébrales sur des images IRM (Imagerie par Résonance Magnétique). Il utilise des techniques de traitement d'image pour délimiter les régions cérébrales, segmenter les lésions, et calculer leur volume et proportion dans différentes structures cérébrales.

## Installation

Prérequis : Python 3.x, Numpy, Matplotlib, Nibabel

Pour installer le projet, suivez ces étapes :

Installez les dépendances nécessaires :
   ```bash
   pip install numpy matplotlib nibabel
   ```

## Utilisation

Pour utiliser ce projet, exécutez les scripts Python dans l'ordre suivant :

1. `mon_module.py` : Ce script contient des fonctions utilitaires pour le traitement d'image.
2. `process_image_irm.py` : Utilisez ce script pour le traitement des images IRM en 3 dimentions.
3. `exploration_irm_lesion.py` : Ce script sert à l'analyse détaillée des image en 2 dimentions et à l'affichage graphique (exploration des données).

## Fonctionnalités

- Délimitation de la région du cerveau dans l’image IRM T1.
- Segmentation de la lésion cérébrale.
- Calcul du nombre de voxels de la lésion dans différentes structures cérébrales.
- Détermination de la proportion de la lésion dans les aires corticales.

## Auteurs

Fraysse Yoann
Célia Changenot
Fanny Tixier
Tom Lebrun

## Contact

Fraysse Yoann - yofraysse@gmail.com
