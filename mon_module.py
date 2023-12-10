#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk


"""

mon_module.py : Module contenant des fonctions pour l'inspection, la manipulation et le traitement d'images en niveaux de gris.

Ce module contient les fonctions suivantes :
- inspection(image) : Fonction d'inspection d'une image.
- affiche(image) : Affiche une image en niveaux de gris.
- affiche_histogramme(image) : Affiche l'histogramme d'une image en niveaux de gris.
- expansion_histogramme(image) : Effectue une expansion de l'histogramme d'une image en niveaux de gris.
- make_noise(image) : Ajoute du bruit à une image en niveaux de gris.
- filtrage(image, k) : Applique un filtrage spatial à une image en niveaux de gris.
- seuillage_fourchette(image, smin, smax) : Applique un seuillage par fourchette à une image en niveaux de gris.
- affiche_binaire(image) : Affiche une image binaire.
- dilatation(image) : Applique une dilatation à une image binaire.
- erosion(image) : Applique une erosion à une image binaire.
- ouverture(image) : Applique une ouverture à une image binaire.
- fermeture(image) : Applique une fermeture à une image binaire.
- correction_nan_bool(image) : Corrige les pixels blancs qui devraient être noir
- affiche_2_image(image1, image2, titre1="Image couleur", titre2="Image binaire") : Créer un affichage de deux images en couleur ou en binaire
- image_couleur(image, couleur) : Crée une image en couleur en remplaçant les pixels de valeur 1 dans l'image d'origine par la couleur spécifiée.

"""

def inspecte(image):
    """ 

    Fonction d'inspection d'une image.
    Cette fonction affiche plusieurs informations sur l'image donnée en entrée.
    
    Paramètres :
        image (numpy.ndarray) : L'image à inspecter.
        
    Affiche :
        - Les dimensions de l'image.
        - La valeur minimale de l'image.
        - La valeur maximale de l'image.
        - Le type de données de l'image.

    """

    print("dimension:", image.shape)  
    print("min:", image.min()) 
    print("max:", image.max())
    print("type de données:", image.dtype, "\n")

def affiche(image, titre="image"):
    """

    Affiche une image en niveaux de gris.
    Cette fonction prend une image en niveaux de gris en entrée et l'affiche dans une fenêtre graphique.
    
    Paramètres :
        image (numpy.ndarray) : L'image à afficher.

    Affiche :
        Une représentation graphique de l'image en niveaux de gris.

    """

    plt.figure()
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title(titre)
    plt.axis('off')
    plt.show()

def affiche_histogramme(image):
    """

    Affiche l'histogramme d'une image en niveaux de gris.
    Cette fonction prend une image en niveaux de gris en entrée, calcule son histogramme, et l'affiche sous forme de graphique.

    Paramètres :
        image (numpy.ndarray) : L'image dont l'histogramme doit être affiché.

    Affiche :
        Un graphique représentant l'histogramme de l'image en niveaux de gris.

    """

    plt.figure()
    plt.hist(image.flatten(), 256)
    plt.xlim([0,255])
    plt.ylim([0,650])
    plt.show()

def expansion_histogramme(image):
    """

    Effectue une expansion de l'histogramme d'une image en niveaux de gris.

    Cette fonction prend une image en niveaux de gris en entrée, effectue une expansion de l'histogramme,
    et renvoie une nouvelle image avec un histogramme étendu sur toute la plage [0, 255].

    Paramètres :
        image (numpy.ndarray) : L'image dont l'histogramme doit être étendu.

    Retourne :
        numpy.ndarray : Une nouvelle image en niveaux de gris avec un histogramme étendu.
    
    """
    
    image_expention = np.float64(image) # conversion de l'image en float64
    
    if np.sum(image) != 0: # Evite les divitions par 0 et donc les messages warning
        image_expention = np.round((255 * (image_expention - image_expention.min()))/(image_expention.max() - image_expention.min())) # calcul de la formule
    image_expention = np.uint8(image_expention) # conversion en uint8

    return image_expention

def make_noise(image):
    """

    Ajoute du bruit à une image en niveaux de gris.

    Cette fonction prend une image en niveaux de gris en entrée, génère un bruit uniforme et l'ajoute à l'image.
    
    Paramètres :
        image (numpy.ndarray) : L'image à laquelle le bruit doit être ajouté.

    Retourne :
        numpy.ndarray : Une nouvelle image en niveaux de gris avec du bruit ajouté.

    """

    # On crée un bruit uniforme
    bruit = np.round(255 * np.random.rand(image.shape[0], image.shape[1]) * .2)

    # On ajoute le bruit à l'image
    image_bruite = np.uint8(np.clip(image + bruit, 0, 255))
    
    return image_bruite

def filtrage(image, k):
    """

    Applique un filtrage spatial à une image en niveaux de gris.

    Cette fonction effectue un filtrage spatial en utilisant un noyau spécifié sur une image en niveaux de gris.
    
    Paramètres :
        image (numpy.ndarray) : L'image à filtrer.
        k (numpy.ndarray) : Le noyau du filtre à appliquer.

    Retourne :
        numpy.ndarray : Une nouvelle image filtrée.

    """
    M, N = image.shape # M (nombre de lignes) et N (nombre de colonnes)
    m, n = k.shape # m (nombre de lignes du noyau) et n (nombre de colonnes du noyau)
    demi_larg_kernel = np.int64(np.ceil(m/2)) # Rayon du noyau

    Imf = np.float64(image)
    for i in range(demi_larg_kernel - 1, M - demi_larg_kernel + 1): # parcours les pixels de l'image
        for j in range(demi_larg_kernel - 1, N-demi_larg_kernel + 1):
            tmp = 0
            for p in range(0, m): # parcours les pixels du noyau
                for q in range(0, n):
                    tmp = tmp + np.float64(image[i - demi_larg_kernel + p, j - demi_larg_kernel + q] * k[p, q])
            Imf[i, j] = tmp

    return np.uint8(np.clip(Imf, 0, 255))

def seuillage_fourchette(image, smin, smax):
    """

    Applique un seuillage par fourchette à une image en Boolean.

    Cette fonction effectue un seuillage par fourchette sur une image en niveaux de gris.
    
    Paramètres :
        image (numpy.ndarray) : L'image à seuiller.
        smin (int) : La valeur minimale du seuil.
        smax (int) : La valeur maximale du seuil.

    Retourne :
        numpy.ndarray : Une nouvelle image seuillée.

    """
    
    image_seuille = np.zeros(image.shape, dtype=bool) # Initialise un tableau de zéros avec la même forme que 'image'
    image_seuille[(image >= smin) & (image <= smax)] = True
    
    return image_seuille

def affiche_binaire(image, titre="image"):
    """

    Affiche une image binaire.

    Cette fonction prend une image binaire en entrée et l'affiche dans une fenêtre graphique.
    
    Paramètres :
        image (numpy.ndarray) : L'image à afficher.

    Affiche :
        Une représentation graphique de l'image binaire.

    """

    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(titre)
    plt.axis('off')
    plt.show()

def dilatation(image):
    """

    Applique une dilatation à une image binaire.

    Cette fonction effectue une dilatation sur une image binaire.
    
    Paramètres :
        image (numpy.ndarray) : L'image à dilater.

    Retourne :
        numpy.ndarray : Une nouvelle image dilatée.

    """

    image_dilate = np.zeros(image.shape, dtype='uint8')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 1:
                image_dilate[i, j] = 1
                if i > 0:
                    image_dilate[i-1, j] = 1
                if i < image.shape[0]-1:
                    image_dilate[i+1, j] = 1
                if j > 0:
                    image_dilate[i, j-1] = 1
                if j < image.shape[1]-1:
                    image_dilate[i, j+1] = 1

    return np.bool_(image_dilate)

def erosion(image):
    """

    Applique une erosion à une image binaire.

    Cette fonction effectue une erosion sur une image binaire.
    
    Paramètres :
        image (numpy.ndarray) : L'image à eroder.

    Retourne :
        numpy.ndarray : Une nouvelle image erodée.
    
    """
    
    image_erosion = np.zeros(image.shape, dtype='uint8')

    for i in range(image.shape[0] - 1):
        for j in range(image.shape[1] - 1):
            if image[i, j] == 1 and image[i - 1, j] == 1 and image[i + 1, j] == 1 and image[i, j - 1] == 1 and image[i, j + 1] == 1:
                image_erosion[i, j] = 1

    return np.bool_(image_erosion)

def ouverture(image):
    """

    Applique une ouverture à une image binaire.

    Cette fonction effectue une ouverture sur une image binaire.
    
    Paramètres :
        image (numpy.ndarray) : L'image à ouvrir.

    Retourne :
        numpy.ndarray : Une nouvelle image ouverte.

    """

    return dilatation(erosion(image))

def fermeture(image):
    """

    Applique une fermeture à une image binaire.

    Cette fonction effectue une fermeture sur une image binaire.
    
    Paramètres :
        image (numpy.ndarray) : L'image à fermer.

    Retourne :
        numpy.ndarray : Une nouvelle image fermée.
    
    """

    return erosion(dilatation(image))

def correction_nan_bool(image):
    """

    Corrige les pixels nan d'une image en noir
    
    """

    tableau_bool = np.full(image.shape, False, dtype=bool)
    tableau_bool[image == 1] = True  # Met à jour les valeurs où image == 1 ou nan
    
    return tableau_bool

def affiche_2_image(image1, image2, titre1="Image couleur", titre2="Image binaire"):
    
    """

    Créer un affichage de deux images en couleur ou en binaire

    :param image1: image
    :type image1: numpy.ndarray
    :param titre1: Le titre de l'image.
    :type titre1: str
    :param image2: image
    :type image2: numpy.ndarray
    :param titre2: Le titre de l'image.
    :type titre2: str
    :return: Un affichage de deux images.
    :rtype: None

    """

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.title(titre1)
    if image1.shape == 3:
        plt.imshow(image1)
    else:
        plt.imshow(image1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(titre2)
    if image2.shape == 3:
        plt.imshow(image2)
    else:
        plt.imshow(image2, cmap='gray')
    plt.axis('off')

    plt.show()

def image_couleur(image, couleur):
    """

    Crée une image en couleur en remplaçant les pixels de valeur 1 dans l'image d'origine par la couleur spécifiée.

    :param image: L'image en niveaux de gris où les pixels de valeur 1 seront colorés.
    :type image: numpy.ndarray
    :param couleur: La couleur à utiliser pour colorer les pixels (choix entre "rouge" et "bleu").
    :type couleur: str
    :return: Une image en couleur avec les pixels colorés selon la couleur spécifiée.
    :rtype: numpy.ndarray
    
    """

    hauteur, largeur = image.shape

    if couleur == "rouge":
        image_couleur = np.zeros((hauteur, largeur, 3), dtype='uint8')
        for i in range(hauteur):
            for j in range(largeur):
                if image.dtype == bool:
                    if image[i, j] == 1:
                        image_couleur[i, j] = (255, 0, 0)
                else:
                    if image[i, j] > 0.0011:
                        image_couleur[i, j] = (255*image[i, j], 0, 0)
   
    elif couleur == "bleu":
        image_couleur = np.zeros((hauteur, largeur, 3), dtype='uint8')
        for i in range(hauteur):
            for j in range(largeur):
                if image.dtype == bool:
                    if image[i, j] == 1:
                        image_couleur[i, j] = (0, 0, 255)
                else:
                    if image[i, j] > 0.0011:
                        image_couleur[i, j] = (0, 0, 255*image[i, j])
    
    elif couleur == "blanc":
        image_couleur = np.zeros((hauteur, largeur, 3), dtype='uint8')
        for i in range(hauteur):
            for j in range(largeur):
                if image.dtype == bool:
                    if image[i, j] == 1:
                        image_couleur[i, j] = (255, 255, 255)
                else:
                    if image[i, j] > 0.0011:
                        image_couleur[i, j] = (255, 255, 255*image[i, j])

    return image_couleur


if __name__ == "__main__":
    print("mon_module.py")