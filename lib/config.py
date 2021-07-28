DENSE_DIMS = (200, 50,)

# The dropout probability
DENSE_DROPOUT = 0.1

# How many epochs we use for training
N_TRAIN_EPOCHS = 20

#image size is resized to
DATASET_IMAGE_SIZE = (512, 512)

IMAGE_SIZE = (256, 512)

#we split an image into halves as training and testing
SPLIT_RATIO = 0.5

N_IMAGE_CHANNELS = 3

BATCH_SIZE = 16

# for resnet, the input size should be larger than 32x32
# model input is 64x64
INPUT_SHAPE = (64, 64, N_IMAGE_CHANNELS)

# Output size for latent space
OUTPUT_DIM = 10
