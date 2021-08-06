from osgeo import gdal
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def get_image(raster_path):
    ds = gdal.Open(raster_path)
    image = np.empty(shape=(ds.RasterYSize, ds.RasterXSize, ds.RasterCount),
                     dtype=np.float32)
    for b in range(ds.RasterCount):
        band = ds.GetRasterBand(b+1).ReadAsArray()
        image[:, :, b] = band
    if image.shape[-1] > 1:
        image = norma_data(image, norma_methods='min-max')
    return image


def norma_data(data, norma_methods="z-score"):
    arr = np.empty(data.shape, dtype=np.float32)
    for i in range(data.shape[-1]):
        array = data[:, :, i]
        mi, ma, mean, std = np.percentile(array, 1), np.percentile(array, 99), array.mean(), array.std()
        if norma_methods == "z-score":
            new_array = (array-mean)/std
        else:
            new_array = (2*(array-mi)/(ma-mi)).clip(0, 1)
        arr[:, :, i] = new_array
    return arr


def write_geotiff(name, prediction, original_path):
    ds = gdal.Open(original_path)
    geo = ds.GetGeoTransform()
    proj = ds.GetProjectionRef()

    driver = gdal.GetDriverByName('GTiff')
    rows, cols, b = prediction.shape
    dataset = driver.Create(name, cols, rows, b, gdal.GDT_Float32)
    dataset.SetGeoTransform(geo)
    dataset.SetProjection(proj)
    for b1 in range(b):
        band = dataset.GetRasterBand(b1+1)
        band.WriteArray(prediction[:, :, b1])


def plot_mask(result):

    arr_2d = result
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    plt.imshow(arr_3d)


palette = {0: (255, 255, 255),  # White
           6: (0, 191, 255),  # DeepSkyBlue
           1: (34, 139, 34),  # ForestGreen
           3: (255, 0, 255),  # Magenta
           2: (0, 255, 0),  # Lime
           5: (255, 127, 80),  # Coral
           4: (255, 0, 0),  # Red
           7: (0, 255, 255),  # Cyan
           8: (0, 255, 0),  # Lime
           9: (0, 128, 128), # other
           }

if __name__ == '__main__':
    path = 'image/'
    # img_2020 = cv.imread(path + '2020.tif', flags=cv.IMREAD_LOAD_GDAL)
    # cv.imshow('2020', img_2020)
    image_2020 = get_image(raster_path=path + 'mask_28.tif')
    print(image_2020.shape)
    plot_mask(image_2020[:, :, 0])
    plt.show()
