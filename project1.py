#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %      
% TDImagen PROYECTO 1: SEGMENTACIÓN DE IMÁGENES                         %
%                                                                       %
% Plantilla para implementar la función principal del sistema,          %
% 'skin_lesion_segmentation', que recibe como entrada la ruta a una     %
% imagen de una lesión y, a su salida, proporciona una máscara de       %
% segmentación, predicha a partir de la solución propuesta.             %
%                                                                       %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import numpy as np
from skimage import io, color, filters
from sklearn.metrics import jaccard_score
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage.morphology import erosion



def skin_lesion_segmentation(img_root):
    
    # El siguiente código implementa el BASELINE incluido en el challenge de
    # Kaggle. 
    
    "PREPROCESADO"

    image = io.imread(img_root)
    image_preprocesado = erosion(image) #erosión
    image_gray = color.rgb2gray(image_preprocesado) #de color a escala de grises
    filtro_gaussian = filters.gaussian(image_gray, 7) #filtro gaussiano con sigma=7
    
#    for i in range(20,500,20):
#        margen=0
#        ventana= filtro_gaussian[:i,:i]
#        threshold = 0.07
#        media = np.mean(ventana)
#        if(media>threshold):
#            margen = i
#            break
#    
#    #recorte de la imagen
#    M,N = filtro_gaussian.shape
#    image_crop = filtro_gaussian[margen: M-margen, margen: N-margen] #imagen recortada
#    m,n = image_crop.shape
#    mascara_final = np.zeros((M,N))
#    margen_x = int((M/2)-(m/2))
#    margen_y = int((N/2)-(n/2))
#    mascara_final[margen_x:(M-margen_x), margen_y: (N-margen_y)] = image_crop #imagen redimensionada   
    
    "SEGMENTACION POR KMEANS"
 
    M,N = filtro_gaussian.shape
    x = np.reshape(filtro_gaussian,(M*N,1))
    K=2
    clustering = KMeans(n_clusters = K, init='k-means++', n_init=10).fit(x)
    L = clustering.labels_
    predicted_mask = np.reshape(L,(M,N))
     
      
    "POSTPROCESADO"
 
    mask = ndimage.morphology.binary_dilation(predicted_mask) #dilatación
    post_predicted_mask = ndimage.morphology.binary_closing(mask) #cierre
    
   
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return post_predicted_mask
        
def evaluate_masks(img_roots, gt_masks_roots):
    """ EVALUATE_MASKS: Función que, dadas dos listas, una con las rutas
        a las imágenes a evaluar y otra con sus máscaras Ground-Truth (GT)
        correspondientes, determina el Mean Average Precision a diferentes
        umbrales de Intersección sobre Unión (IoU) para todo el conjunto de
        imágenes.
    """
    score = []
    for i in np.arange(np.size(img_roots)):
        predicted_mask = skin_lesion_segmentation(img_roots[i])
        gt_mask = io.imread(gt_masks_roots[i])/255     
        score.append(jaccard_score(np.ndarray.flatten(gt_mask),np.ndarray.flatten(predicted_mask)))
    mean_score = np.mean(score)
    print('Jaccard Score sobre el conjunto de imágenes proporcionado: '+str(mean_score))
    return mean_score
