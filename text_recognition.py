import time

import cv2
import numpy as np

img1 = cv2.imread('bin/DIGNE.png')
img2 = cv2.imread('bin/Drive.png')
blank = cv2.imread('blank.png')
blank = cv2.resize(blank, (25, 50), interpolation=cv2.INTER_AREA)
print(img1.shape)
t = time.time()
vis = np.concatenate((cv2.resize(img1, (100, 50), interpolation=cv2.INTER_AREA),
                      blank,
                      cv2.resize(img2, (100, 50), interpolation=cv2.INTER_AREA)), axis=1)
print(time.time() - t)
cv2.imwrite('out.png', vis)
