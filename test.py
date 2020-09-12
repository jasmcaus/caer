import cv2 as cv

file = r'C:\Users\aus\Downloads\atd-sdop-izKxBS9s5vs-unsplash.jpg'

img = cv.imread(file)
img = cv.resize(img, (500,500))
cv.imshow('img', img)
# (b,g,r) = cv.split(img.astype('float32'))

# b -= 50
# g -= 50
# r -= 50

# cv.imshow('ee', cv.merge([b,g,r]))

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Imga', img)

img = img.astype('float32')

img -= cv.mean(img)[0]
# print('Mean', cv.mean(img))

cv.imshow('Subtracted', img)

cv.waitKey(0)