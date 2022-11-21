import cv2
import pytesseract
import os
import numpy as np
import keras_ocr
import matplotlib.pyplot as plt

# create an array of filenames from all the images in the current folder
images = [img for img in os.listdir(".") if img.endswith(".jpeg")]

# KERAS OCR
#pipeline = keras_ocr.pipeline.Pipeline()
#
## loop through all the images in the current folder
#for image in images:
#    # recognize text with keras-ocr and output the result to a text file called "keras.txt"
#    prediction_groups = pipeline.recognize([image])
#    with open("keras.txt", "a") as f:
#        f.write(image + "\n")
#        for prediction in prediction_groups[0]:
#            f.write(prediction[0] + " ")
#        f.write("\n\n\n")
#    


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# PYTESSERACT OCR

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 


# foreach image in the current folder, run OCR on it
for image in images:
    # read the image
    img = cv2.imread(image)

    # preprocess the image
    gray = get_grayscale(img)
    thresh = thresholding(gray)
    #opening = opening(thresh)
    #canny = canny(gray)

    # run OCR on the image
    text = pytesseract.image_to_string(gray, lang='fra')

    
    # put the title of the image without the extension and the text extracted from it into a text file
    with open(os.path.splitext(image)[0] + ".txt", "w", encoding='utf-8') as f:
        f.write(image + """

        """)

        

        f.write(text + """


        """)

    # print the text to the console
    print(text)