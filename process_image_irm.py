#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
import os
from tqdm import tqdm
import mon_module as mm


plt.close('all')

"""

Ouverture de toutes les images

"""

# Ouverture de toutes les images du dossier
dossier_base = "./images"     
fichiers = os.listdir(dossier_base)
chemins_des_fichiers = [os.path.join(dossier_base, fichier) for fichier in fichiers]

# Ouverture image IRM brute
irm_load = nb.load(chemins_des_fichiers[2])
irm_data = irm_load.get_fdata()
nx, ny, nz = irm_data.shape
header = irm_load.header
pixdim = header['pixdim']
resolution_spatiale = pixdim[1:4]

# Ouverture du masque
mask_load = nb.load(chemins_des_fichiers[3])
mask_data = mask_load.get_fdata()

#Structures cérébrales
# Cortex / Lesion -- vérifier si on  doit etandre l'histogramme
cortex_load = nb.load(chemins_des_fichiers[11])
cortex_data = cortex_load.get_fdata()

# Substance blanche / Lesion
substance_blanche_load = nb.load(chemins_des_fichiers[7])
substance_blanche_data = substance_blanche_load.get_fdata()

# Noyaux gris centraux / Lesion
noyaux_gris_centraux_load = nb.load(chemins_des_fichiers[1])
noyaux_gris_centraux_data = noyaux_gris_centraux_load.get_fdata()

#Aires corticale
# Cortex insulaire / Lesion
cortex_insulaire_load = nb.load(chemins_des_fichiers[8])
cortex_insulaire_data = cortex_insulaire_load.get_fdata()

# Cortex préfrontal / Lesion
cortex_prefrontal_load = nb.load(chemins_des_fichiers[0])
cortex_prefrontal_data = cortex_prefrontal_load.get_fdata()

# Cortex pariétal / Lesion
cortex_parietal_load = nb.load(chemins_des_fichiers[6])
cortex_parietal_data = cortex_parietal_load.get_fdata()

# Cortex occipital / Lesion
cortex_occipital_load = nb.load(chemins_des_fichiers[4])
cortex_occipital_data = cortex_occipital_load.get_fdata()

# Cortex temporal / Lesion
cortex_temporal_load = nb.load(chemins_des_fichiers[9])
cortex_temporal_data = cortex_temporal_load.get_fdata()

# Cortex limbique / Lesion
cortex_limbique_load = nb.load(chemins_des_fichiers[5])
cortex_limbique_data = cortex_limbique_load.get_fdata()

# Cortex moteur / Lesion
cortex_moteur_load = nb.load(chemins_des_fichiers[10])
cortex_moteur_data = cortex_moteur_load.get_fdata()

"""

Traitement de toutes les données en 3D

"""

# Calcule du volume de la lésion
sum_pixel_lesion, sum_pixel_cortex_lesion, sum_pixel_sb_lesion, sum_pixel_ngc_lesion = 0, 0, 0, 0
sum_surface_lesion_insulaire, sum_surface_cortex_insulaire, sum_surface_lesion_prefrontal, sum_surface_cortex_prefrontal = 0, 0, 0, 0
sum_surface_lesion_parietal, sum_surface_cortex_parietal, sum_surface_lesion_occipital, sum_surface_cortex_occipital = 0, 0, 0, 0
sum_surface_lesion_temporal, sum_surface_cortex_temporal, sum_surface_lesion_limbique, sum_surface_cortex_limbique = 0, 0, 0, 0
sum_surface_lesion_moteur, sum_surface_cortex_moteur, sum_pixel_cortex, sum_pixel_sb, sum_pixel_ngc = 0, 0, 0, 0, 0

seuil_proba = 0.0011
resolution_spatiale_isotrope = resolution_spatiale[0] * resolution_spatiale[1] * resolution_spatiale[2] #0.35 # mm3
for z in tqdm(range(nz), desc = "Traitement des données"):
    # Extraction IRM coupe z
    irm = irm_data[:, :, z]
    irm = mm.expansion_histogramme(irm) # retourne une image en uint8

    # Extraction masque coupe z
    mask = mask_data[:, :, z] 
    mask_corriger = mm.correction_nan_bool(mask) 
    
    # Création d'une nouvelle image avec seulement le cerveau sans la boite craniène (en theorie)
    cerveau = irm * mask_corriger # Cerveau isolé
    cerveau = mm.expansion_histogramme(cerveau)

    # Segmentation de la lésion
    lesion_brute = mm.seuillage_fourchette(cerveau, 1, 30) 
    lesion = mm.ouverture(lesion_brute) 
    sum_pixel_lesion += np.sum(lesion) 

    # Cortex / Lesion
    cortex = cortex_data[:, :, z] 
    cortex = mm.correction_nan_bool(cortex)
    sum_pixel_cortex += np.sum(cortex)
    cortex_lesion = lesion * cortex
    sum_pixel_cortex_lesion += np.sum(cortex_lesion)
    # Substance blanche / Lesion
    substance_blanche = substance_blanche_data[:, :, z]
    substance_blanche = mm.correction_nan_bool(substance_blanche)
    sum_pixel_sb += np.sum(substance_blanche)
    substance_blanche_lesion = lesion * substance_blanche
    sum_pixel_sb_lesion += np.sum(substance_blanche_lesion)
    # Noyaux gris centraux / Lesion
    noyaux_gris_centraux = noyaux_gris_centraux_data[:, :, z]
    noyaux_gris_centraux = mm.correction_nan_bool(noyaux_gris_centraux)
    sum_pixel_ngc += np.sum(noyaux_gris_centraux)
    noyaux_gris_centraux_lesion = lesion * noyaux_gris_centraux
    sum_pixel_ngc_lesion += np.sum(noyaux_gris_centraux_lesion)

    # Cortex insulaire / Lesion
    cortex_insulaire = cortex_insulaire_data[:, :, z]
    cortex_insulaire_lesion = lesion * cortex_insulaire
    sum_surface_lesion_insulaire += np.sum(cortex_insulaire_lesion[cortex_insulaire_lesion > seuil_proba]) 
    sum_surface_cortex_insulaire += np.sum(cortex_insulaire[cortex_insulaire > seuil_proba])
    # Cortex préfrontal / Lesion
    cortex_prefrontal = cortex_prefrontal_data[:, :, z]
    cortex_prefrontal_lesion = lesion * cortex_prefrontal
    sum_surface_lesion_prefrontal += np.sum(cortex_prefrontal_lesion[cortex_prefrontal_lesion > seuil_proba])
    sum_surface_cortex_prefrontal += np.sum(cortex_prefrontal[cortex_prefrontal > seuil_proba])
    # Cortex pariétal / Lesion
    cortex_parietal = cortex_parietal_data[:, :, z]
    cortex_parietal_lesion = lesion * cortex_parietal
    sum_surface_lesion_parietal += np.sum(cortex_parietal_lesion[cortex_parietal_lesion > seuil_proba])
    sum_surface_cortex_parietal += np.sum(cortex_parietal[cortex_parietal > seuil_proba])
    # Cortex occipital / Lesion
    cortex_occipital = cortex_occipital_data[:, :, z]
    cortex_occipital_lesion = lesion * cortex_occipital
    sum_surface_lesion_occipital += np.sum(cortex_occipital_lesion[cortex_occipital_lesion > seuil_proba])
    sum_surface_cortex_occipital += np.sum(cortex_occipital[cortex_occipital > seuil_proba])
    # Cortex temporal / Lesion
    cortex_temporal = cortex_temporal_data[:, :, z]
    cortex_temporal_lesion = lesion * cortex_temporal
    sum_surface_lesion_temporal += np.sum(cortex_temporal_lesion[cortex_temporal_lesion > seuil_proba])
    sum_surface_cortex_temporal += np.sum(cortex_temporal[cortex_temporal > seuil_proba])
    # Cortex limbique / Lesion
    cortex_limbique = cortex_limbique_data[:, :, z]
    cortex_limbique_lesion = lesion * cortex_limbique
    sum_surface_lesion_limbique += np.sum(cortex_limbique_lesion[cortex_limbique_lesion > seuil_proba])
    sum_surface_cortex_limbique += np.sum(cortex_limbique[cortex_limbique > seuil_proba])
    # Cortex moteur / Lesion
    cortex_moteur = cortex_moteur_data[:, :, z]
    cortex_moteur_lesion = lesion * cortex_moteur
    sum_surface_lesion_moteur += np.sum(cortex_moteur_lesion[cortex_moteur_lesion > seuil_proba])
    sum_surface_cortex_moteur += np.sum(cortex_moteur[cortex_moteur > seuil_proba])


"""

Nombre de Voxels dans la Lésion : Supposons que nous avons compté N voxels dans la lésion.
Volume d'un Voxel Unique : Chaque voxel a un volume de 0.35×0.35×0.35 mm³, puisque la résolution est de 0.35 mm dans chaque dimension.
Calcul du Volume Total de la Lésion : Multipliez le nombre de voxels de la lésion par le volume d'un voxel unique pour obtenir le volume total de la lésion.
Volume de la Lésion = N × (0.35 × 0.35 × 0.35)mm3

"""
print("\nLe volume de la lésion est de :", round(sum_pixel_lesion * resolution_spatiale_isotrope, 2), "mm3 soit", sum_pixel_lesion, "voxels.\n") # 7053

print("Le volume du cortex est de :", round(sum_pixel_cortex * resolution_spatiale_isotrope, 2), "mm3 soit", sum_pixel_cortex, "voxels.")
print("Le volume de la lésion dans le cortex est de :", round(sum_pixel_cortex_lesion * resolution_spatiale_isotrope, 2), "mm3 soit", round(sum_pixel_cortex_lesion / sum_pixel_lesion * 100, 2), "% de la lésion et", sum_pixel_cortex_lesion, "voxels.\n") # 3553 - 50.38  %

print("Le volume de la substance blanche est de :", round(sum_pixel_sb * resolution_spatiale_isotrope, 2), "mm3 soit", sum_pixel_sb, "voxels.")
print("Le volume de la lésion dans la substance blanche est de :", round(sum_pixel_sb_lesion * resolution_spatiale_isotrope, 2), "mm3 soit", round(sum_pixel_sb_lesion / sum_pixel_lesion * 100, 2), "% de la lésion et", sum_pixel_sb_lesion, "voxels.\n") # 1819 - 25.79  %

print("Le volume des noyaux gris centraux est de :", round(sum_pixel_ngc * resolution_spatiale_isotrope, 2), "mm3 soit", sum_pixel_ngc, "voxels.")
print("Le volume de la lésion dans les noyaux gris centraux est de :", round(sum_pixel_ngc_lesion * resolution_spatiale_isotrope, 2), "mm3 soit", round(sum_pixel_ngc_lesion / sum_pixel_lesion * 100, 2), "% de la lésion et", sum_pixel_ngc_lesion, "voxels.\n") # 743 - 10.53  %

print("## La lésion dans le cortex insulaire ##")
print("Le volume du cortex insulaire est de", round(sum_surface_cortex_insulaire * resolution_spatiale_isotrope, 2), "mm3 soit", round(sum_surface_cortex_insulaire, 2), "voxels.")
print("Le volume de la lésion dans le cortex insulaire est de", round(sum_surface_lesion_insulaire * resolution_spatiale_isotrope, 2), "mm3 soit", round(sum_surface_lesion_insulaire, 2), "voxels.")
print("La proportion de la lésion dans le cortex insulaire est de", round(sum_surface_lesion_insulaire / sum_pixel_lesion * 100, 2), "%\n")

print("## La lésion dans le cortex préfrontal ##")
print("Le volume du cortex préfrontal est de", round(sum_surface_cortex_prefrontal * resolution_spatiale_isotrope, 2), "mm3 soit", round(sum_surface_cortex_prefrontal, 2), "voxels.")
print("Le volume de la lésion dans le cortex préfrontal est de", round(sum_surface_lesion_prefrontal * resolution_spatiale_isotrope, 2), "mm3 soit ", round(sum_surface_lesion_prefrontal, 2), "voxels.")
print("La proportion de la lésion dans le cortex préfrontal est de", round(sum_surface_lesion_prefrontal / sum_pixel_lesion * 100, 2), "%\n")

print("## La lésion dans le cortex pariétal ##")
print("Le volume du cortex pariétal est de", round(sum_surface_cortex_parietal * resolution_spatiale_isotrope, 2), "mm3 soit", round(sum_surface_cortex_parietal, 2), "voxels.")
print("Le volume de la lésion dans le cortex pariétal est de", round(sum_surface_lesion_parietal * resolution_spatiale_isotrope, 2), "mm3 soit", round(sum_surface_lesion_parietal, 2), "voxels.")
print("La proportion de la lésion dans le cortex pariétal est de", round(sum_surface_lesion_parietal / sum_pixel_lesion * 100, 2), "%\n")

print("## La lésion dans le cortex occipital ##")
print("Le volume du cortex occipital est de", round(sum_surface_cortex_occipital * resolution_spatiale_isotrope, 2), "mm3 soit", round(sum_surface_cortex_occipital, 2), "voxels.")
print("Le volume de la lésion dans le cortex occipital est de", round(sum_surface_lesion_occipital * resolution_spatiale_isotrope, 2), "mm3 soit", round(sum_surface_lesion_occipital, 2), "voxels.")
print("La proportion de la lésion dans le cortex occipital est de", round(sum_surface_lesion_occipital / sum_pixel_lesion * 100, 2), "%\n")

print("## La lésion dans le cortex temporal ##")
print("Le volume du cortex temporal est de", round(sum_surface_cortex_temporal * resolution_spatiale_isotrope, 2), "mm3 soit", round(sum_surface_cortex_temporal, 2), "voxels.")
print("Le volume de la lésion dans le cortex temporal est de", round(sum_surface_lesion_temporal * resolution_spatiale_isotrope, 2), "mm3 soit", round(sum_surface_lesion_temporal, 2), "voxels.")
print("La proportion de la lésion dans le cortex temporal est de", round(sum_surface_lesion_temporal / sum_pixel_lesion * 100, 2), "%\n")

print("## La lésion dans le cortex limbique ##")
print("Le volume du cortex limbique est de", round(sum_surface_cortex_limbique * resolution_spatiale_isotrope, 2), "mm3 soit", round(sum_surface_cortex_limbique, 2), "voxels.")
print("Le volume de la lésion dans le cortex limbique est de", round(sum_surface_lesion_limbique * resolution_spatiale_isotrope, 2), "mm3 soit", round(sum_surface_lesion_limbique, 2), "voxels.")
print("La proportion de la lésion dans le cortex limbique est de", round(sum_surface_lesion_limbique / sum_pixel_lesion * 100, 2), "%\n")

print("## La lésion dans le cortex moteur ##")
print("Le volume du cortex moteur est de", round(sum_surface_cortex_moteur * resolution_spatiale_isotrope, 2), "mm3 soit", round(sum_surface_cortex_moteur, 2), "voxels.")
print("Le volume de la lésion dans le cortex moteur est de", round(sum_surface_lesion_moteur * resolution_spatiale_isotrope, 2), "mm3 soit", round(sum_surface_lesion_moteur, 2), "voxels.")
print("La proportion de la lésion dans le cortex moteur est de", round(sum_surface_lesion_moteur / sum_pixel_lesion * 100, 2), "%\n")
