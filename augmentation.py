import cv2
import numpy as np
import sys
import os
import random
import shutil
from tqdm.auto import tqdm

OUTPUTS_NUM = 500
INPUT_FOLDER = "flags/"
BORDER_SIZE = 20

def add_noise(image, mean=0, std=0.5):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

try:
    shutil.rmtree("augmented_output")
except:
    pass
os.mkdir("augmented_output")
os.mkdir("augmented_output/train/")
os.mkdir("augmented_output/test/")
countries = os.listdir(INPUT_FOLDER)
for j in tqdm(countries):
    os.mkdir("augmented_output/train/" + j)
    os.mkdir("augmented_output/test/" + j)
    source_images = os.listdir(INPUT_FOLDER + j)
    for i in range(OUTPUTS_NUM):
        num = random.choice(source_images)
        # if j == "Austria" and num == "Austria.bmp":     #for debug only to test changes on specific image -> austria/9.bmp
        img = cv2.imread(INPUT_FOLDER + j + "/" + num)
        img = cv2.resize(img, (127,127))       
        img = cv2.copyMakeBorder(img, np.random.randint(BORDER_SIZE), np.random.randint(BORDER_SIZE), np.random.randint(BORDER_SIZE), np.random.randint(BORDER_SIZE), cv2.BORDER_REFLECT) 
        img = cv2.resize(img, (127,127))       
        # AUGMENTATION OPERATIONS THEMSELF
        flip = np.random.randint(10)
        if flip == 0:
            img = cv2.flip(img, np.random.randint(2))
        degree = np.random.randint(21) - 10
        radian = np.absolute(degree/360*2*np.pi)
        # print(degree, radian)
        a = 64 * np.tan(radian)/(1 + np.tan(radian))
        b = 64 - a
        c = np.sqrt(a*a + b*b)
        zoom = 1 + np.random.random() * 0.4
        # print(a, b, c)
        # print(img.shape)
        # add gausian noise
        img = add_noise(img)
        rotate = cv2.getRotationMatrix2D((img.shape[0]//2, img.shape[0]//2), degree, 64/c)
        img = cv2.warpAffine(img, rotate, img.shape[:2])
        max_trans = (zoom - 1) * img.shape[0] // 2
        if max_trans > 0:
            shiftx = np.random.randint(max_trans*2) - max_trans
            shifty = np.random.randint(max_trans*2) - max_trans
        else:
            shiftx = 0
            shifty = 0
        zoom_m = cv2.getRotationMatrix2D((img.shape[0]//2, img.shape[0]//2), 0, zoom)
        zoom_m[0][2] += shiftx
        zoom_m[1][2] += shifty
        img = cv2.warpAffine(img, zoom_m, img.shape[:2])
        if i < (OUTPUTS_NUM * 0.8):
            cv2.imwrite("augmented_output/train/" + j + "/augmented_" + str(i) + ".bmp", img)
        else:
            cv2.imwrite("augmented_output/test/" + j + "/augmented_" + str(i) + ".bmp", img)