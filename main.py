import project1 as project
import test_csv_for_kaggle as csv_test
import matplotlib.pyplot as plt
#from skimage import io
#leer imágenes y máscaras
import os
from skimage import io

#CARGAMOS LAS IMÁGENES Y MÁSCARAS DE TRAIN
data_dir= 'C:/Users/María/Desktop/UC3M/Proyecto'

train_imgs_files = [os.path.join(data_dir,'train/images',f) for f in sorted(os.listdir(os.path.join(data_dir,'train/images'))) 
            if (os.path.isfile(os.path.join(data_dir,'train/images',f)) and f.endswith('.jpg'))]

train_masks_files = [os.path.join(data_dir,'train/masks',f) for f in sorted(os.listdir(os.path.join(data_dir,'train/masks'))) 
            if (os.path.isfile(os.path.join(data_dir,'train/masks',f)) and f.endswith('.png'))]

#Ordenamos para que cada imagen se corresponda con cada máscara
train_imgs_files.sort()
train_masks_files.sort()
print("Número de imágenes de train", len(train_imgs_files))
print("Número de máscaras de train", len(train_masks_files))



#CARGAMOS LAS IMÁGENES Y MÁSCARAS DE TEST
test_imgs_files = [os.path.join(data_dir,'test/images',f) for f in sorted(os.listdir(os.path.join(data_dir,'test/images'))) 
            if (os.path.isfile(os.path.join(data_dir,'test/images',f)) and f.endswith('.jpg'))]

test_masks_files = [ os.path.join(data_dir,'test/masks',f) for f in sorted(os.listdir(os.path.join(data_dir,'test/masks'))) 
            if (os.path.isfile(os.path.join(data_dir,'test/masks',f)) and f.endswith('.png')) ]

test_imgs_files.sort()
test_masks_files.sort()
print("Número de imágenes de test", len(test_imgs_files))
print("Número de máscaras de test", len(test_masks_files))



#-----------------------------------

img_roots = train_imgs_files.copy()
gt_masks_roots = train_masks_files.copy()
mean_score = project.evaluate_masks(img_roots, gt_masks_roots)
#mean_score_test = project.evaluate_masks(test_imgs_files,test_masks_files)




#Una vez satisfechos con el resultado, generamos el fichero para hacer la submission en Kaggle
dir_images_name = 'C:/Users/María/Desktop/UC3M/Proyecto/test/images'
csv_name='test_prediction_erosion_rgb2gray_gaussian_kmean_fill_holes_dilation_opening.csv'
csv_test.test_prediction_csv(dir_images_name, csv_name)

