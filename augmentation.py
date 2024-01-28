import cv2
import numpy as np
import sys
import os
import random
import shutil
from tqdm.auto import tqdm
import math

OUTPUTS_NUM = 1000
INPUT_FOLDER = "flags/"
BORDER_SIZE = 10
WAVE = 6

def add_noise(image, mean=0, std=0.5):
    std = np.random.random() * 0.5
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def add_wave(image, amplitude=0):
    rows, cols = image.shape[:2]
    img_output = np.zeros((rows - amplitude, cols - amplitude, 3), dtype=image.dtype)
    coef_x = np.random.randint(1,3)
    coef_y = np.random.randint(1,3)
    for i in range(rows-amplitude): 
        for j in range(cols-amplitude): 
            offset_x = int(amplitude * math.sin(coef_x * 3.14 * i / 180)) 
            offset_y = int(amplitude * math.cos(coef_y * 3.14 * j / 180)) 
            if i+offset_y < rows and j+offset_x < cols: 
                img_output[i,j] = image[(i+offset_y)%rows,(j+offset_x)%cols] 
    return img_output

def color_normalizer(image):
    WHITE = [255, 255, 255]
    BLACK = [0, 0, 0]
    RED = [0, 0, 255]
    GREEN = [0, 100, 0]
    BLUE_LIGHT = [255, 100, 100]
    BLUE_DARK = [150, 0, 0]
    YELLOW = [0, 255, 255]
    colors = [WHITE, BLACK, RED, GREEN, BLUE_LIGHT, BLUE_DARK, YELLOW]
    rows, cols = image.shape[:2]
    img_output = np.zeros((rows, cols, 3), dtype=image.dtype)
    for i in range(rows): 
        for j in range(cols): 
            pick = []
            metrics = []
            pixel = image[i,j]
            for color in colors:
                loss = [0, 0, 0]
                for k in range(3):
                    loss[k] = abs(color[k] - pixel[k])
                pick.append(loss)
            for l in pick:
                metrics.append(sum(l))
            img_output[i,j] = colors[metrics.index(min(metrics))]
                # if image[i,j][k] > 128:
                #     img_output[i,j][k] = 255
                # else:
                #     img_output[i,j][k] = 0

    return img_output

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
    for i in source_images:
        img = cv2.imread(INPUT_FOLDER + j + "/" + i)
        img = color_normalizer(img)
        cv2.imwrite(INPUT_FOLDER + j + "/" + i, img)
        for i in range(OUTPUTS_NUM):
            num = random.choice(source_images)
            # if j == "Austria" and num == "Austria.bmp":     #for debug only to test changes on specific image -> austria/9.bmp
            img = cv2.imread(INPUT_FOLDER + j + "/" + num)
            img = cv2.resize(img, (64,64))
            border = np.random.randint(WAVE)
            img = cv2.copyMakeBorder(img, np.random.randint(BORDER_SIZE), border + np.random.randint(BORDER_SIZE), np.random.randint(BORDER_SIZE), border + np.random.randint(BORDER_SIZE), cv2.BORDER_REFLECT) 
            img = add_wave(img, border)
            img = cv2.resize(img, (64,64))
            # print(f"iteration {i}\nzoom_additional {zoom_additional}\namplitude")
            # AUGMENTATION OPERATIONS THEMSELF
            flip = np.random.randint(100)
            if flip == 0:
                img = cv2.flip(img, np.random.randint(2))
            flip = np.random.randint(100)
            if flip == 0:
                img = cv2.rotate(img, cv2.ROTATE_180)
            degree = np.random.randint(15) - 7
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