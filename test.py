import cv2
import numpy as np
img =np.int8(cv2.imread("F:\merge.png"))

img1 =np.int8(cv2.imread("F:\dst.png"))



img2 = img - img1

b = img2[:,:,1]

a = 9