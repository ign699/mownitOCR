import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from os import listdir

def read_img_to_grey(name):
    text = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    text = 255 - text
    return text



def show_img(img):
    cv2.imshow('title', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_contours(dialted_text):
    cnts = cv2.findContours(dialted_text, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    return cnts[1]


def dilate_words(text, width, height):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, height))
    return cv2.dilate(text, rectKernel, 1)


def print_with_contours(text, contour):
    text = cv2.cvtColor(text, cv2.COLOR_GRAY2RGB)
    show_img(cv2.drawContours(text, [contour], 0, (0,255,0), 1))

letters = read_img_to_grey("abc.png")
text = cv2.imread("abc.png", cv2.IMREAD_GRAYSCALE)
contours = list(reversed(get_contours(dilate_words(letters, 3, 8))))


i=97
for contour in contours:
    print_with_contours(text, contour)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.imwrite(str(i) + ".png", text[y: y + h, x: x + w])
    i+=1


