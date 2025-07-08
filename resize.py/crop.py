import os

import cv2


img = cv2.imread(os.path.join('.', 'dogs.jpg'))

print(img.shape)

cropped_img = img[130:230,150:290]

cv2.imshow('img', img)
cv2.imshow('cropped_img', cropped_img)
cv2.waitKey(0)