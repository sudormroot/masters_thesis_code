import numpy as np
import pandas as pd
import os
import sys 
import imageio

libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "lib" 
sys.path.append(libpath)

from texture_dataset import load_texture_dataset
from imgaug_model import build_augmentation_model


def create_dataset( *,
                    subdir = "Multiband_Brodatz_Texture",
                    dataset_root = "datasets_new",
                    image_size = (512, 512),
                    n_channels = 3,
                    output_shape = (128, 128),
                    n_images_per_class = 256,
                    rand_seed = 1921
                    ):

    dataset_path = dataset_root + os.path.sep +  subdir
    output_shape = (*output_shape, n_channels)

    print("dataset_path = ", dataset_path)
    print("output_shape = ", output_shape)
    print("n_images_per_class = ", n_images_per_class)

    X, y = load_texture_dataset(image_size = image_size, subdir = subdir)

    imgaug = build_augmentation_model(  input_shape = (*image_size, n_channels),
                                        output_shape = output_shape,
                                        rand_seed = rand_seed
                                        )


    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    for yi in y:
        #print(yi)
        yi_path = dataset_path + os.path.sep + yi
        if not os.path.exists(yi_path):
            os.makedirs(yi_path)


    for i, (Xi, yi) in enumerate(zip(X, y)):
        yi_path = dataset_path + os.path.sep + yi
        print(yi)
        Xi = np.expand_dims(Xi, axis = 0)
        for j in range(n_images_per_class):
            Xij = imgaug(Xi)
            Xij = Xij[0]
            Xij = np.array(Xij, dtype = np.uint8)
            Xij_name = yi_path + os.path.sep + f"{yi}_{j}.png"
            print(Xij_name)
            imageio.imwrite(Xij_name, Xij)



    print()
    print("Completed!")
    print()


if __name__ == "__main__":

    subdir = "Multiband_Brodatz_Texture"

    if len(sys.argv) == 2:
        subdir = sys.argv[1]

    create_dataset( 
                    subdir = subdir,
                    dataset_root = "datasets_new",
                    image_size = (512, 512),
                    n_channels = 3,
                    output_shape = (128, 128),
                    n_images_per_class = 256
                  )


