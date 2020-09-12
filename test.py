import os
import cv2 as cv

count = 0
DIR = r'F:\Datasets\Dogs vs Cats Kaggle (Full)\Dogs and Cats\Train'
meanb, meang, meanr = 0,0,0

for root, _, files in os.walk(DIR):
    for file in files:
        if file.endswith('.png') or file.endswith('.jpg'):
            count += 1
            filepath = root + os.sep + file
            img = cv.imread(filepath)
            b,g,r = cv.mean(img)[:3]

            meanb += b
            meang += g
            meanr += r
            break
        break

print(count)
meanb /= count
meang /= count
meanr /= count
    

print(meanb, meang, meanr)