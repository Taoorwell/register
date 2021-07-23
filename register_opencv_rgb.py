import numpy as np
import cv2 as cv

img_1_path = 'image/skull.png'
img_2_path = 'image/skull_1.png'

img_1 = cv.imread(img_1_path)
img_2 = cv.imread(img_2_path)

# There is no differences between using rgb and using gray.
# img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
# img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

# Step 1: extract key points and descriptions
akaze = cv.AKAZE_create()
# akaze = cv.SIFT_create()
# akaze = cv.ORB_create(50)
kp_1, descriptor_1 = akaze.detectAndCompute(img_1, None)
kp_2, descriptor_2 = akaze.detectAndCompute(img_2, None)
# print(kp_1[0].pt)l.

# show the keypoints and images
# img_1 = cv.drawKeypoints(img_1, kp_1, img_1)
img_11 = cv.drawKeypoints(image=img_1, keypoints=kp_1, outImage=img_1)
cv.imshow(winname='image', mat=img_11)
cv.waitKey(0)
# cv.destroyWindow(winname='image')
# plt.imshow(img_11)
# plt.show()

img_22 = cv.drawKeypoints(image=img_2, keypoints=kp_2, outImage=img_2)
cv.imshow(winname='image', mat=img_22)
cv.waitKey(0)
# cv.destroyWindow(winname='image')
# plt.imshow(img_22)
# plt.show()

# Step 2: using the descriptions to match key points and filter good match points.
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(descriptor_1, descriptor_2, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
# print(good_matches[0].queryIdx)
# print(len(good_matches))

# Draw matches
img_3 = cv.drawMatches(img_1, kp_1, img_2, kp_2, good_matches, None,
                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow(winname='img_3', mat=img_3)
cv.waitKey(0)
# cv.destroyWindow(winname='img_3')
# plt.figure(figsize=(20, 10))
# plt.imshow(img_3)
# plt.show()

# Select good matched key points
ref_matched_kpts = [kp_1[m.queryIdx].pt for m in good_matches]
sensed_matched_kpts = [kp_2[m.trainIdx].pt for m in good_matches]

# Step 3: compute homography and use it to warp image.
# Compute homography, shape (3, 3)
H, status = cv.findHomography(np.array(sensed_matched_kpts), np.array(ref_matched_kpts),
                              cv.RANSAC, 5.0)

# Warp image
warped_image = cv.warpPerspective(img_2, H, (img_2.shape[1], img_2.shape[0]))
cv.imshow(winname='warped_image', mat=warped_image)
cv.waitKey(0)
cv.destroyAllWindows()
# plt.imshow(warped_image)
# plt.show()