import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import neurite as ne
from utility import get_image, creat_patches
from matplotlib import pyplot as plt


def image_pair_datasets():
    pass


if __name__ == '__main__':
    patch_size, stride = 256, 128
    inshape = (patch_size, patch_size)

    nb_features = [
        [32, 32, 32, 32],
        [32, 32, 32, 32, 32, 16]
    ]

    # image_path = [r'image/po_97258_pan_0000000.tif', r'image/po_97258_pan_0010000.tif']
    image_path = [r'image/2016_11_north_urban_0.tif', r'image/2020_11_north_urban_0.tif']
    image_0 = get_image(image_path[0], norma=True)
    image_1 = get_image(image_path[1], norma=True)
    image_0, image_1 = image_0[:, :, 0:1], image_1[:, :, 0:1]
    patches_gen = creat_patches(image_0, image_1, patch_size, stride)
    patches = patches_gen()
    print(len(list(patches)))
    # patches_gen_1 = creat_patches(image_1, patch_size, stride)
    # patches_gen = make_gen_callable(patches_gen)
    # patches = list(patches_gen)
    # print(patches[0].shape)
    datasets = tf.data.Dataset.from_generator(patches_gen,
                                              output_types=(tf.float32, tf.float32,
                                                            tf.float32, tf.float32),
                                              output_shapes=((patch_size, patch_size, 1),
                                                             (patch_size, patch_size, 1),
                                                             (patch_size, patch_size, 1),
                                                             (patch_size, patch_size, 1)))
    datasets = datasets.batch(8)

    def map_function(x0, x1, x2, x3):
        return (x0, x1), (x2, x3)
    datasets = datasets.map(map_function).repeat()

    # for i, j in datasets:
    #     print(i[0].shape, i[1].shape, j[0].shape, j[1].shape)
    # print('Length of datasets:{}'.format(len(datasets)))

    # Model Import
    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
    vxm_model.summary()
    # Model Prediction
    # vxm_model.load_weights(r'image/register-2020')
    # for i, _ in datasets:
    #     a, b = vxm_model.predict(i)
    #     plt.subplot(141)
    #     plt.imshow(i[0][2, :, :, 0])
    #
    #     plt.subplot(142)
    #     plt.imshow(i[1][2, :, :, 0])
    #
    #     plt.subplot(143)
    #     plt.imshow(a[2, :, :, 0])
    #
    #     plt.subplot(144)
    #     plt.imshow(b[2, :, :, 0])
    #     # ne.plot.flow([b[0, :, :].squeeze()], width=1)
    #
    #     plt.show()
    #     break

    # Loss function
    losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    lambda_param = 0.05
    loss_weights = [1, lambda_param]

    # Model Compile
    vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

    # Train model
    hist = vxm_model.fit_generator(datasets,
                                   epochs=10,
                                   steps_per_epoch=50,
                                   verbose=1)
    vxm_model.save_weights(filepath=r'image/register-2020')

    # for i in datasets:
    #     print(i[0].shape, i[1].shape)
    #     plt.subplot(121)
    #     plt.imshow(i[0][9, :, :, 0])
    #
    #     plt.subplot(122)
    #     plt.imshow(i[1][9, :, :, 0])
    #
    #     plt.show()
    #     break

    # datasets = tf.data.Dataset.from_generator(patches_gen_0,
    #                                           output_types=tf.float32,
    #                                           output_shapes=(patch_size, patch_size, 1))
    # datasets = datasets.batch(10)
    # for i in datasets:
    #     print(i.shape)
    # print(list(patches_gen))
