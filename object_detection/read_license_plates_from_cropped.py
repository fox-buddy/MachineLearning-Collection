import glob
import cv2
import numpy as np
import random
import easyocr
import imutils
from matplotlib import pyplot as plt

path_to_read = "./LicensePlateImages/*"
file_list = images_to_progress = glob.glob(path_to_read)

test_file = file_list[4]

print(test_file)

# will be file_list in further iterations
paths_of_files_to_Progess = [test_file]


# !!! Takes about 3 Seconds to read license plate


for idx_file, val_file in enumerate(paths_of_files_to_Progess):

    # A) Convert to Greyscale --> colors like a piece of paper
    img = cv2.imread(val_file)  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    print("Grayscaled")

    #) F) Extract Text
    reader = easyocr.Reader(['de'])
    result = reader.readtext(bfilter)

    print(result)

    # extracted_text = result[0][1]

    # G) Format Extracted text
    # trimmed_plate = extracted_text.replace(" ", "")
    # print(f"License plate is {trimmed_plate}")