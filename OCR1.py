import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from os import listdir

def read_img_to_grey(name):
    text = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    text = 255 - text
    return text

def read_letter(name):
    letter = cv2.imread("./sans_serif/" + name, cv2.IMREAD_GRAYSCALE)
    letter = 255 - letter
    contour = get_contours(letter)
    x, y, w, h = cv2.boundingRect(contour[0])
    img = letter[y: y + h, x: x + w]
    return img

def show_img(img):
    cv2.imshow('title', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def dilate_words(text, width):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 5))
    return cv2.dilate(text, rectKernel, 1)

def get_contours(dialted_text):
    cnts = cv2.findContours(dialted_text, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    return cnts[1]

def get_img_inside_contour(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return [x, y, w, h]

def get_letters_corr(text, letter):
    text_fft = np.fft.fft2(text)
    pattern_fft = np.fft.fft2(np.rot90(letter, 2), text_fft.shape)
    m = np.multiply(text_fft, pattern_fft)
    corr = np.fft.ifft2(m)
    corr = np.abs(corr)
    corr = corr.astype(float)
    corr[corr < 0.9 * np.amax(corr)] = 0
    return corr

def coor_to_letter(corr):
    coors = []
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if corr[i][j] > 0:
                coors.append(((i, j), corr[i][j]))
    return coors

def check_if_in_word(contour, letter, x, y):
    a, b, w, h = get_img_inside_contour(contour)
    return (letter[0] > y and letter[0] < y + h and letter[1] > x and letter[1] < x + w)


def plot_coor(corr):
    plt.imshow(corr, cmap='jet')
    plt.show()

def get_letters():
    return listdir('./sans_serif')

def generate_dict_for_letters(text, letters):
    letters_dict = {}
    for letter in letters:
        letters_dict[letter[:-4]] = coor_to_letter(get_letters_corr(text, read_letter(letter)))
    return letters_dict

def print_with_contours(text, contour):
    text = cv2.cvtColor(text, cv2.COLOR_GRAY2RGB)
    show_img(cv2.drawContours(text, [contour], 0, (0,255,0), 1))

def get_letters_dict():
    letters = get_letters()
    print(letters)
    return generate_dict_for_letters(text, letters)

def find_matching_letter(letter_contour, letters_dict):
    a, b, c, d = get_img_inside_contour(letter_contour)
    best_corr = 0
    best_letter = ""
    pos = 0
    for letter in letters_coor_dict:
        letter_positions = letters_coor_dict[letter]
        correlation = 0
        poss = 0
        for position in letter_positions:
            if check_if_in_word(letter_contour, position[0], x+ a, y+ b):
                if position[1] > correlation:
                    correlation = position[1]
                    poss = position[0][1]
        if correlation > best_corr:
            best_letter = letter
            pos = poss
            best_corr = correlation
    return best_letter, pos

def is_word_in_line(word, line):
    x_l, y_l, w_l, h_l = cv2.boundingRect(line)
    x, y, w, h = cv2.boundingRect(word)
    return y >= y_l and y <= y_l + h_l




image = "test.png"
text = read_img_to_grey(image)
letters_coor_dict = get_letters_dict()


words = list(reversed(get_contours(dilate_words(text, 7))))
lines = list(reversed(get_contours(dilate_words(text, 30))))
whole_text = ""
text = read_img_to_grey(image)

for line in lines:
    for word in words:
        x, y, w, h = cv2.boundingRect(word)
        letters = []
        for letter_contour in get_contours(text[y: y + h, x: x + w]):
            char, pos = find_matching_letter(letter_contour, letters_coor_dict)
            letters.append((char, pos))
        if is_word_in_line(word, line):
            letters.sort(key=lambda tup: tup[1])
            for letter in letters:
                whole_text += letter[0]
            else:
                whole_text += " "
    whole_text += '\n'

print(whole_text)
