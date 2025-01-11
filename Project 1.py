from tkinter import *

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, Conv2D

from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.utils import plot_model

from IPython.display import SVG, Image

import tkinter as tk

from tkinter import filedialog

from PIL import ImageTk, Image

from PIL import Image, ImageDraw, ImageFont

import os

from tkinter import messagebox

from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

import cv2

import numpy as np

from skimage.color import rgb2gray

from scipy import ndimage

img_size = 48

batch_size = 64

datagen_train = ImageDataGenerator(horizontal_flip=True)

train_generator = datagen_train.flow_from_directory("data//trainset//", target_size=
(48, 48), batch_size=batch_size, class_mode='categorical', shuffle=True)

datagen_validation = ImageDataGenerator(horizontal_flip=True)

validation_generator = datagen_train.flow_from_directory("data//testset//", target_size=
(48, 48), batch_size=batch_size, class_mode='categorical', shuffle=True)

window = Tk()

window.geometry("1200x600")

window.title("Uwais_Mansuri")

window.config(background="brown")

window.iconbitmap(default='main.ico')

li = Label(window, text='Uwais_SkinDisease_CNN', font='cambria 22 bold italic', bg="brown", fg="yellow")

li.place(x=50, y=0)


def Open():
    global file_open

    file_open = filedialog.askopenfilename(title='Choose image', filetypes=[("Image File", '.jpg')])

    return file_open


def ipim():
    x = Open()

    global file_open

    global my_image

    global imglabel

    img = Image.open(x)

    img = img.resize((225, 225), Image.ANTIALIAS)

    img = ImageTk.PhotoImage(img)

    my_image = img

    panel = Label(window, image=img)

    panel.image = img

    imglabel = Label(window, image=img)

    imglabel.place(x=150, y=70)


def process():
    global test_img, output

    global file_open

    global my_image

    global opt1

    global imglabel1

    global imglabel2

    image_name_input = file_open

    img = Image.open(image_name_input)

    img = img.resize((225, 225), Image.ANTIALIAS)

    image_name_output = 'IP.jpg'

    img.save(image_name_output)

    img1 = ImageTk.PhotoImage(Image.open("IP.jpg"))

    img11 = cv2.imread(r'IP.jpg', 1)

    gray = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('GS.jpg', gray)

    img = Image.open('GS.jpg')

    img = img.resize((225, 225), Image.ANTIALIAS)

    img = ImageTk.PhotoImage(img)

    panel = Label(window, image=img)

    panel.image = img

    imglabel1 = Label(window, image=img)

    imglabel1.place(x=530, y=70)

    ret, thresh = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV +
                                cv2.THRESH_OTSU)
    cv2.imwrite('SG.jpg', thresh)

    (T, thresh) = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)

    (T, threshInv) = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY_INV)

    cv2.imwrite('DR.jpg', threshInv)

    img = Image.open('SG.jpg')

    img = img.resize((225, 225), Image.ANTIALIAS)

    img = ImageTk.PhotoImage(img)

    panel = Label(window, image=img)

    panel.image = img

    imglabel2 = Label(window, image=img)

    imglabel2.place(x=150, y=340)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    I = os.path.getsize(image_name_output)

    panel = Label(window, image=img1)

    panel.image = img1

    classifier = load_model('data//trainset//Model.h5')

    test_img = image.load_img(file_open, target_size=(48, 48))

    test_img = image.img_to_array(test_img)

    test_img = np.expand_dims(test_img, axis=0)

    result = classifier.predict(test_img)

    a = result.argmax()

    s = train_generator.class_indices

    name = []

    for i in s:
        name.append(i)

    for i in range(len(s)):
        if (i == a):
            p = name[i]
            output = name[i]

    opt1 = tk.Label(window, text=output, fg='yellow', bg='saddle brown', font='cambria 19 bold')

    opt1 = tk.Label(window, text=output, fg='yellow', bg='saddle brown', font='cambria 19 bold')

    opt1.place(x=520, y=480)


def canny(img11, sigma=0.33):
    v = np.median(img11)

    lower = int(max(0, (1.0 - sigma) * v))

    upper = int(min(255, (1.0 + sigma) * v))

    edged = cv2.Canny(img11, lower, upper)

    return edged


def grosp():
    global opt1

    global imglabel

    global imglabel1

    global imglabel2

    global imglabel3

    opt1.config(text="")

    imglabel.config(image="")

    imglabel1.config(image="")

    imglabel2.config(image="")


bi1 = Button(window, text='BROWSE IMAGE', width=14, borderwidth=1, bg='cyan', fg='gray1', command=ipim,
             font='cambria 17 bold italic')

bi1.place(x=880, y=140)

bi2 = Button(window, text='DETECT DISEASE', borderwidth=1, width=14, bg='cyan', fg='gray1', command=process,
             font='cambria 17 bold italic')

bi2.place(x=880, y=240)

bi3 = Button(window, text='RESET', borderwidth=1, bg='cyan', fg='gray1', width=14, command=grosp,
             font='cambria 17 bold italic')

bi3.place(x=880, y=340)

bi4 = Button(window, text='EXIT', borderwidth=1, bg='cyan', fg='gray1', width=14, command=window.destroy,
             font='cambria 17 bold italic')

bi4.place(x=880, y=440)

l3 = Label(window, text='Input Image', fg='white', bg='saddle brown', font='cambria 15 bold italic')

l3.place(x=200, y=300)

l4 = Label(window, text='Preprocessing', fg='white', bg='saddle brown', font='cambria 15 bold italic')

l4.place(x=580, y=300)

l5 = Label(window, text='Detection', fg='white', bg='saddle brown', font='cambria 15 bold italic')

l5.place(x=210, y=570)

l7 = Label(window, text='***************************', fg='yellow', bg='saddle brown', font='cambria 15 bold italic')

l7.place(x=510, y=410)

li5 = Label(window, text='>>> MENU <<<', fg='white', bg='saddle brown', font='cambria 16 bold italic')

li5.place(x=900, y=100)

l6 = Label(window, text='DETECTED DISEASE', fg='yellow', bg='saddle brown', font='cambria 19 bold italic')

l6.place(x=520, y=430)

l9 = Label(window, text='***************************', fg='yellow', bg='saddle brown', font='cambria 15 bold italic')

l9.place(x=510, y=460)

window.mainloop()
