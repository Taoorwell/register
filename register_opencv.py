import numpy as np
import matplotlib.pyplot as plt
from utility import get_image, write_geotiff
import cv2 as cv

img_1_path = 'image/2016.tif'
img_2_path = 'image/2020.tif'

# img_1 = cv.imread(img_1_path)
# img_2 = cv.imread(img_2_path)


def float32_uint8(x):
    x = x*255
    return x.astype(np.uint8)


img_1 = get_image(img_1_path)
img_2 = get_image(img_2_path)

img_1 = img_1[:, :, [1, 2, 3]]
img_2 = img_2[:, :, [1, 2, 3]]

# both opencv and matplotlib can show numpy.float32.
# plt.imshow(img_1)
# plt.show()
# cv.imshow(winname='..', mat=img_1)
# cv.waitKey(0)

# Step 1: extract key points and descriptions
akaze = cv.AKAZE_create()
# akaze = cv.SIFT_create()
kp_1, descriptor_1 = akaze.detectAndCompute(img_1, None)
kp_2, descriptor_2 = akaze.detectAndCompute(img_2, None)
# print(kp_1[0].pt)l.

# show the keypoints and images
# img_1 = cv.drawKeypoints(img_1, kp_1, img_1)
image_1 = float32_uint8(img_1)
img_11 = cv.drawKeypoints(image=image_1, keypoints=kp_1, outImage=image_1.copy())
# cv.imshow(winname='image', mat=img_1)
plt.imshow(img_11)
plt.show()

image_2 = float32_uint8(img_2)
img_22 = cv.drawKeypoints(image=image_2, keypoints=kp_1, outImage=image_2.copy())
# cv.imshow(winname='image', mat=img_1)
plt.imshow(img_22)
plt.show()

# Step 2: using the descriptions to match key points and filter good match points.
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(descriptor_1, descriptor_2, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.60 * n.distance:
        good_matches.append(m)
# print(good_matches[0].queryIdx)
# print(len(good_matches))

# Draw matches
img_3 = cv.drawMatches(image_1, kp_1, image_2, kp_2, good_matches, None,
                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(20, 10))
plt.imshow(img_3)
plt.show()

# Select good matched key points
ref_matched_kpts = [kp_1[m.queryIdx].pt for m in good_matches]
sensed_matched_kpts = [kp_2[m.trainIdx].pt for m in good_matches]

# Step 3: compute homography and warp image.
# Compute homography
H, status = cv.findHomography(np.array(sensed_matched_kpts), np.array(ref_matched_kpts),
                              cv.RANSAC, 5.0)

# Warp image
warped_image = cv.warpPerspective(img_2, H, (img_2.shape[1], img_2.shape[0]))
plt.imshow(warped_image)
plt.show()
# write_geotiff('image/2020_wrap.tif', warped_image[:, :, [2, 1, 0]], 'image/2020.tif')
