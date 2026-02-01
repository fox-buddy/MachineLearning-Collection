import glob
import cv2
import numpy as np
import random
import easyocr
import imutils
from matplotlib import pyplot as plt

path_to_read = "./LicensePlateImages/*"
file_list = images_to_progress = glob.glob(path_to_read)

test_file = file_list[3]

print(test_file)

# will be file_list in further iterations
paths_of_files_to_Progess = [test_file]


# !!! Takes about 3 Seconds to read license plate


for idx_file, val_file in enumerate(paths_of_files_to_Progess):

    # A) Convert to Greyscale --> Makes it easier to detect edges (plates)
    img = cv2.imread(val_file)  
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))                # plot
    plt.title('Original Image')                                     # plot
    plt.show()                                                      # plot
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    plt.imshow(cv2.cvtColor(bfilter, cv2.COLOR_BGR2RGB))            # plot
    plt.title('Processed Image')                                    # plot
    plt.show()                                                      # plot

    # B) Detect edges
    # Notice that these numbers are lower and upper threshold values
    # We have to perform some tests to find the very best values. Or adjust them over the time. 30-200 or 50-150 should be fine to start
    # Edges will be white
    edged_img = cv2.Canny(bfilter, 30, 200);
    plt.imshow(cv2.cvtColor(edged_img, cv2.COLOR_BGR2RGB))          # plot
    plt.title("Edge Detection")                                     # plot
    plt.show()                                                      # plot

    # C) Find Contours
    keypoints = cv2.findContours(edged_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    print("Location: ", location)

    # D) Marking Number Plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)  
    new_image = cv2.bitwise_and(img, img, mask=mask)  
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))          # plot
    plt.title('Masked Image')                                       # plot
    plt.show()                                                      # plot

    #) E) Crop Image
    (x, y) = np.where(mask == 255)  
    (x1, y1) = (np.min(x), np.min(y)) 
    (x2, y2) = (np.max(x), np.max(y))  
    cropped_image = gray[x1:x2+1, y1:y2+1]
    # This is where OCR will take place after YOLO model croped. But cropped image should at least be grayscaled. That should make it much easier
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))      # plot
    plt.title('Cropped Image')                                      # plot
    plt.show()                                                      # plot

    #) F) Extract Text
    reader = easyocr.Reader(['de'])
    result = reader.readtext(cropped_image)

    print(result)

    # extracted_text = result[0][1]

    # G) Format Extracted text
    # trimmed_plate = extracted_text.replace(" ", "")
    # print(f"License plate is {trimmed_plate}")