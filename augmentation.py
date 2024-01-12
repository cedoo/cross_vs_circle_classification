import cv2
import numpy as np
import sys

for j in ["crosses/", "circles/"]:
    count = 11
    for i in range(count, 5001):
        picture = np.random.randint(10) + 1
        img = cv2.imread(j + str(picture) + '.bmp')
        flip = np.random.randint(2)
        if flip == 0:
            img = cv2.flip(img, 0)

        rotate_side = np.random.randint(4)
        if rotate_side == 0:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotate_side == 1:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotate_side == 2:
            img = cv2.rotate(img, cv2.ROTATE_180)

        degree = np.random.randint(21)
        degree -= 15
        # degree = int(sys.argv[1])
        radian = np.absolute(degree/360*2*np.pi)
        zoom = np.random.random() - 0.7
        a = 64 * np.tan(radian)/(1 + np.tan(radian))
        b = 64 - a
        c = np.sqrt(a*a + b*b)
        rotate = cv2.getRotationMatrix2D((32, 32), degree, 64/c + zoom)
        rotate[0,2] += np.random.randint(20) - 10
        rotate[1,2] += np.random.randint(20) - 10
        rotated = cv2.warpAffine(img, rotate, (64, 64))
        # cv2.imshow("img", rotated)
        # cv2.waitKey(0)
        filename = j + str(count) + ".bmp"
        count += 1
        cv2.imwrite(filename, rotated)