#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
import os

import mon_module as mm

plt.close('all')

"""

Délimitation de la région du cerveau dans l’image T1 par application du masque du cerveau

"""

dossier_base = "./images"    
fichiers = os.listdir(dossier_base)
chemins_des_fichiers = [os.path.join(dossier_base, fichier) for fichier in fichiers]

"""

Choix parmis les fichiers :
[0] = rR_prefrontal.nii
[1] = rgm_subco.nii
[2] = MARM_685_3MOIS_LESION_CORO_T1.nii
[3] = rBrainMask.nii
[4] = rR_occipital.nii
[5] = rR_limbic.nii
[6] = rR_parietal.nii
[7] = rwm.nii
[8] = rR_insular.nii
[9] = rR_temporal.nii
[10] = rR_motor.nii
[11] = rgm_cortex.nii

"""

Z = 69

# Ouverture image IRM brute
irm_load = nb.load(chemins_des_fichiers[2])
irm_data = irm_load.get_fdata()
#mm.inspecte(irm_data) # dimension: (288, 288, 150) / max: 3080 / float64 
nx, ny, nz = irm_data.shape

irm = irm_data[:, :, Z] # Chargement de l'image
irm = mm.expansion_histogramme(irm) # et on fait une expansion d'histogramme
irm = np.rot90(irm, k=3) # on fait une rotation de 270°
#mm.affiche(irm, fichiers[2] + "coupe "+ str(68))


# Ouverture du masque
mask_load = nb.load(chemins_des_fichiers[3])
mask_data = mask_load.get_fdata()
mask = mask_data[:, :, Z]
mask = mm.correction_nan_bool(mask)
mask = np.rot90(mask, k=3) 
#mm.affiche_binaire(mask)

# Création d'une nouvelle image avec seulement le cerveau sans la boite craniène (en theorie)
cerveau = irm * mask # Cerveau isolé
cerveau = mm.expansion_histogramme(cerveau)
#mm.affiche(cerveau)

"""

Segmentation de la lésion sur l’image T1 par un seuillage

"""


# Segmentation de la lésion
lesion_brute = mm.seuillage_fourchette(cerveau, 1, 30) # 1
lesion = mm.ouverture(lesion_brute)
#mm.affiche_binaire(lesion)

sum_pixel_lesion = np.sum(lesion)
print("Le nombre de pixel dans la lésion est de :", sum_pixel_lesion, "\n") # 289
mm.affiche_2_image(lesion_brute, lesion, "Seuillage fourchette", "Lésion")
mm.affiche_2_image(mm.image_couleur(irm, "bleu") + mm.image_couleur(mask, "rouge"), cerveau, "IRM (bleu) et le masque (rouge)", "Cerveau isolé")


"""

Détermination du nombre de voxels de la lésion compris dans chacune des structures cérébrales
(cortex, substance blanche, noyaux sous-corticaux)

"""


print("\nDétermination du nombre de voxels de la lésion compris dans chacune des structures cérébrales \n\n")

# Cortex / Lesion 
cortex_load = nb.load(chemins_des_fichiers[11])
cortex_data = cortex_load.get_fdata()
cortex = cortex_data[:, :, Z] 
cortex = np.rot90(cortex, k=3)
cortex = mm.correction_nan_bool(cortex)
cortex_lesion = lesion * cortex
mm.affiche_2_image(mm.image_couleur(cortex, "bleu") + mm.image_couleur(lesion, "rouge"), cortex_lesion, "Cortex (bleu) + Lésion (rouge)", "Cortex * Lésion")
print("Le nombre de pixel de la lésion dans le cortex est de :", np.sum(cortex_lesion), "soit ", round(np.sum(cortex_lesion) / sum_pixel_lesion * 100, 2), "% de la lésion \n")

# Substance blanche / Lesion
substance_blanche_load = nb.load(chemins_des_fichiers[7])
substance_blanche_data = substance_blanche_load.get_fdata()
substance_blanche = substance_blanche_data[:, :, Z]
substance_blanche = np.rot90(substance_blanche, k=3)
substance_blanche = mm.correction_nan_bool(substance_blanche)
substance_blanche_lesion = lesion * substance_blanche
mm.affiche_2_image(mm.image_couleur(substance_blanche, "bleu") + mm.image_couleur(lesion, "rouge"), substance_blanche_lesion, "Substance blanche (bleu) \n+ Lésion (rouge)", "Substance blanche * Lésion")
print("Le nombre de pixel de la lésion dans la substance blanche est de :", np.sum(substance_blanche_lesion), "soit ", round(np.sum(substance_blanche_lesion) / sum_pixel_lesion * 100, 2), "% de la lésion \n")

# Noyaux gris centraux / Lesion
noyaux_gris_centraux_load = nb.load(chemins_des_fichiers[1])
noyaux_gris_centraux_data = noyaux_gris_centraux_load.get_fdata()
noyaux_gris_centraux = noyaux_gris_centraux_data[:, :, Z]
noyaux_gris_centraux = np.rot90(noyaux_gris_centraux, k=3)
noyaux_gris_centraux = mm.correction_nan_bool(noyaux_gris_centraux)
noyaux_gris_centraux_lesion = lesion * noyaux_gris_centraux
mm.affiche_2_image(mm.image_couleur(noyaux_gris_centraux, "bleu") + mm.image_couleur(lesion, "rouge"), noyaux_gris_centraux_lesion, "Noyaux gris centraux (bleu) \n+ Lésion (rouge)", "Noyaux gris centraux * Lésion")
print("Le nombre de pixel de la lésion dans les noyaux gris centraux est de :", np.sum(noyaux_gris_centraux_lesion), "soit ", round(np.sum(noyaux_gris_centraux_lesion) / sum_pixel_lesion * 100, 2), "% de la lésion \n")

"""

Détermination de la proportion de la lésion dans chacune des 7 aires corticales

"""

print("\nDétermination de la proportion de la lésion dans chacune des 7 aires corticales \n\n")

# Cortex insulaire / Lesion -- vérifier si on  doit etandre l'histogramme
print("## La lésion dans le cortex insulaire ##")
cortex_insulaire_load = nb.load(chemins_des_fichiers[8])
cortex_insulaire_data = cortex_insulaire_load.get_fdata()
cortex_insulaire = cortex_insulaire_data[:, :, Z]
cortex_insulaire = np.rot90(cortex_insulaire, k=3)
cortex_insulaire_lesion = lesion * cortex_insulaire # couleur chelou mais bonne valeur

# Donne une mesure de l'étendue totale de la lésion dans chaque région, en tenant compte de l'incertitude ou de la confiance que chaque voxel appartienne à cette région.
surface_lesion_insulaire = np.sum(cortex_insulaire_lesion[cortex_insulaire_lesion > 0.0011]) # mm.somme_pixel_blanc(cortex_insulaire_lesion)
surface_cortex_insulaire = np.sum(cortex_insulaire[cortex_insulaire > 0.0011]) # mm.somme_pixel_blanc(cortex_insulaire)

print("La surface du cortex insulaire est de ", round(surface_cortex_insulaire, 2), "pixels")
print("La surface de la lésion dans le cortex insulaire est de ", round(surface_lesion_insulaire, 2), "pixels")
print("La proportion de la lésion dans le cortex insulaire est de ", round(surface_lesion_insulaire / surface_cortex_insulaire * 100, 2), "%\n")
mm.affiche_2_image(mm.image_couleur(cortex_insulaire, "bleu") + mm.image_couleur(lesion, "rouge"), cortex_insulaire_lesion, "Cortex insulaire (bleu) \n+ Lésion (rouge)", "Cortex insulaire * Lésion")



# Cortex préfrontal / Lesion
print("## La lésion dans le cortex préfrontal ##")
cortex_prefrontal_load = nb.load(chemins_des_fichiers[0])
cortex_prefrontal_data = cortex_prefrontal_load.get_fdata()
cortex_prefrontal = cortex_prefrontal_data[:, :, Z]
cortex_prefrontal = np.rot90(cortex_prefrontal, k=3)
cortex_prefrontal_lesion = lesion * cortex_prefrontal

surface_lesion_prefrontal = mm.somme_pixel_blanc(cortex_prefrontal_lesion) # np.sum(ima[ima > 0.0011])
surface_cortex_prefrontal = mm.somme_pixel_blanc(cortex_prefrontal) # np.sum(ima[ima > 0.0011])

print("La surface du cortex préfrontal est de ", round(surface_cortex_prefrontal, 2), "pixels")
print("La surface de la lésion dans le cortex préfrontal est de ", round(surface_lesion_prefrontal, 2), "pixels")
print("La proportion de la lésion dans le cortex préfrontal est de ", round(surface_lesion_prefrontal / surface_cortex_prefrontal * 100, 2), "%\n")
mm.affiche_2_image(mm.image_couleur(cortex_prefrontal, "bleu") + mm.image_couleur(lesion, "rouge"), cortex_prefrontal_lesion, "Cortex préfrontal (bleu) \n+ Lésion (rouge)", "Cortex préfrontal * Lésion")

# Cortex pariétal / Lesion
print("## La lésion dans le cortex pariétal ##")
cortex_parietal_load = nb.load(chemins_des_fichiers[6])
cortex_parietal_data = cortex_parietal_load.get_fdata()
cortex_parietal = cortex_parietal_data[:, :, Z]
cortex_parietal = np.rot90(cortex_parietal, k=3)
cortex_parietal_lesion = lesion * cortex_parietal

surface_lesion_parietal = mm.somme_pixel_blanc(cortex_parietal_lesion) # np.sum(ima[ima > 0.0011])
surface_cortex_parietal = mm.somme_pixel_blanc(cortex_parietal) # np.sum(ima[ima > 0.0011])

print("La surface du cortex pariétal est de ", round(surface_cortex_parietal, 2), "pixels")
print("La surface de la lésion dans le cortex pariétal est de ", round(surface_lesion_parietal, 2), "pixels")
print("La proportion de la lésion dans le cortex pariétal est de ", round(surface_lesion_parietal / surface_cortex_parietal * 100, 2), "%\n")
mm.affiche_2_image(mm.image_couleur(cortex_parietal, "bleu") + mm.image_couleur(lesion, "rouge"), cortex_parietal_lesion, "Cortex pariétal (bleu) \n+ Lésion (rouge)", "Cortex pariétal * Lésion")

# Cortex occipital / Lesion
print("## La lésion dans le cortex occipital ##")
cortex_occipital_load = nb.load(chemins_des_fichiers[4])
cortex_occipital_data = cortex_occipital_load.get_fdata()
cortex_occipital = cortex_occipital_data[:, :, 100]
cortex_occipital = np.rot90(cortex_occipital, k=3)
cortex_occipital_lesion = lesion * cortex_occipital

surface_lesion_occipital = mm.somme_pixel_blanc(cortex_occipital_lesion) # np.sum(ima[ima > 0.0011])
surface_cortex_occipital = mm.somme_pixel_blanc(cortex_occipital) # np.sum(ima[ima > 0.0011])

#print("La surface du cortex occipital est de ", round(surface_cortex_occipital, 2), "pixels")
#print("La surface de la lésion dans le cortex occipital est de ", round(surface_lesion_occipital, 2), "pixels")
#print("La proportion de la lésion dans le cortex occipital est de ", round(surface_lesion_occipital / surface_cortex_occipital * 100, 2), "%\n")
#mm.affiche_2_image(mm.image_couleur(cortex_occipital, "bleu") + mm.image_couleur(lesion, "rouge"), cortex_occipital_lesion, "Cortex occipital (bleu) \n+ Lésion (rouge)", "Cortex occipital * Lésion")

# Cortex temporal / Lesion
print("## La lésion dans le cortex temporal ##")
cortex_temporal_load = nb.load(chemins_des_fichiers[9])
cortex_temporal_data = cortex_temporal_load.get_fdata()
cortex_temporal = cortex_temporal_data[:, :, Z]
cortex_temporal = np.rot90(cortex_temporal, k=3)
cortex_temporal_lesion = lesion * cortex_temporal

surface_lesion_temporal = mm.somme_pixel_blanc(cortex_temporal_lesion) # np.sum(ima[ima > 0.0011])
surface_cortex_temporal = mm.somme_pixel_blanc(cortex_temporal) # np.sum(ima[ima > 0.0011])

print("La surface du cortex temporal est de ", round(surface_cortex_temporal, 2), "pixels")
print("La surface de la lésion dans le cortex temporal est de ", round(surface_lesion_temporal, 2), "pixels")
print("La proportion de la lésion dans le cortex temporal est de ", round(surface_lesion_temporal / surface_cortex_temporal * 100, 2), "%\n")
mm.affiche_2_image(mm.image_couleur(cortex_temporal, "bleu") + mm.image_couleur(lesion, "rouge"), cortex_temporal_lesion, "Cortex temporal (bleu) \n+ Lésion (rouge)", "Cortex temporal * Lésion")

# Cortex limbique / Lesion
print("## La lésion dans le cortex limbique ##")
cortex_limbique_load = nb.load(chemins_des_fichiers[5])
cortex_limbique_data = cortex_limbique_load.get_fdata()
cortex_limbique = cortex_limbique_data[:, :, Z]
cortex_limbique = np.rot90(cortex_limbique, k=3)
cortex_limbique_lesion = lesion * cortex_limbique

surface_lesion_limbique = mm.somme_pixel_blanc(cortex_limbique_lesion)  # np.sum(ima[ima > 0.0011])
surface_cortex_limbique = mm.somme_pixel_blanc(cortex_limbique) # np.sum(ima[ima > 0.0011])

print("La surface du cortex limbique est de ", round(surface_cortex_limbique, 2), "pixels")
print("La surface de la lésion dans le cortex limbique est de ", round(surface_lesion_limbique, 2), "pixels")
print("La proportion de la lésion dans le cortex limbique est de ", round(surface_lesion_limbique / surface_cortex_limbique * 100, 2), "%\n")
mm.affiche_2_image(mm.image_couleur(cortex_limbique, "bleu") + mm.image_couleur(lesion, "rouge"), cortex_limbique_lesion, "Cortex limbique (bleu) \n+ Lésion (rouge)", "Cortex limbique * Lésion")

# Cortex moteur / Lesion
print("## La lésion dans le cortex moteur ##")
cortex_moteur_load = nb.load(chemins_des_fichiers[10])
cortex_moteur_data = cortex_moteur_load.get_fdata()
cortex_moteur = cortex_moteur_data[:, :, Z]
cortex_moteur = np.rot90(cortex_moteur, k=3)
cortex_moteur_lesion = lesion * cortex_moteur

surface_lesion_moteur = mm.somme_pixel_blanc(cortex_moteur_lesion)  # np.sum(ima[ima > 0.0011])
surface_cortex_moteur = mm.somme_pixel_blanc(cortex_moteur) # np.sum(ima[ima > 0.0011])

print("La surface du cortex moteur est de ", round(surface_cortex_moteur, 2), "pixels")
print("La surface de la lésion dans le cortex moteur est de ", round(surface_lesion_moteur, 2), "pixels")
print("La proportion de la lésion dans le cortex moteur est de ", round(surface_lesion_moteur / surface_cortex_moteur * 100, 2), "%\n")
mm.affiche_2_image(mm.image_couleur(cortex_moteur, "bleu") + mm.image_couleur(lesion, "rouge"), cortex_moteur_lesion, "Cortex moteur (bleu) \n+ Lésion (rouge)", "Cortex moteur * Lésion")