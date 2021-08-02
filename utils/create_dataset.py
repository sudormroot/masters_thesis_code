import numpy as np
import pandas as pd
import os
import sys 
import imageio

libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep + "lib" 
sys.path.append(libpath)

from texture_dataset import load_texture_dataset
from imgaug_model import build_augmentation_model
from split_dataset import split_dataset_by_pixel

def create_dataset_from_xy( *, 
                            X, 
                            y, 
                            dataset_path, 
                            subdir, 
                            imgaug,
                            n_images_per_class):
 
    for yi in y:
        #print(yi)
        yi_path = dataset_path + os.path.sep + subdir + os.path.sep + yi

        if not os.path.exists(yi_path):
            os.makedirs(yi_path)


    for i, (Xi, yi) in enumerate(zip(X, y)):
        yi_path = dataset_path + os.path.sep + subdir + os.path.sep + yi
        print(yi)
        Xi = np.expand_dims(Xi, axis = 0)
        for j in range(n_images_per_class):
            Xij = imgaug(Xi)
            Xij = Xij[0]
            Xij = np.array(Xij, dtype = np.uint8)
            Xij_name = yi_path + os.path.sep + f"{yi}_{j}.png"
            print(Xij_name)
            imageio.imwrite(Xij_name, Xij)


   

def create_dataset( *,
                    dataset_name = "Multiband_Brodatz_Texture",
                    dataset_root = "datasets_new",
                    image_size = (512, 512),
                    n_channels = 3,
                    output_shape = (128, 128),
                    n_images_per_class = 256,
                    rand_seed = 1921
                    ):

    dataset_path = dataset_root + os.path.sep +  dataset_name
    output_shape = (*output_shape, n_channels)

    print("dataset_path = ", dataset_path)
    print("output_shape = ", output_shape)
    print("n_images_per_class = ", n_images_per_class)

    X, y = load_texture_dataset(image_size = image_size, dataset_name = dataset_name)

    X_train, y_train, X_test, y_test = split_dataset_by_pixel(X, y)

    imgaug = build_augmentation_model(  input_shape = (*image_size, n_channels),
                                        output_shape = output_shape,
                                        rand_seed = rand_seed
                                        )

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        os.makedirs(dataset_path + os.path.sep + "train")
        os.makedirs(dataset_path + os.path.sep + "test")

    create_dataset_from_xy( X = X_train, 
                            y = y_train, 
                            dataset_path = dataset_path, 
                            subdir = "train",
                            imgaug = imgaug,
                            n_images_per_class = n_images_per_class)

    create_dataset_from_xy( X = X_test, 
                            y = y_test, 
                            dataset_path = dataset_path, 
                            subdir = "test",
                            imgaug = imgaug,
                            n_images_per_class = n_images_per_class)


    print()
    print("Completed!")
    print()


if __name__ == "__main__":

    dataset_name = "Multiband_Brodatz_Texture"

    if len(sys.argv) == 2:
        dataset_name = sys.argv[1]

    create_dataset( 
                    dataset_name = dataset_name,
                    dataset_root = "datasets_new",
                    image_size = (512, 512),
                    n_channels = 3,
                    output_shape = (128, 128),
                    n_images_per_class = 256
                  )


