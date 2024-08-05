import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from imageprocessing import img_proc
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from imutils import contours
import imutils
from tqdm import tqdm
import os
from tensorflow.keras.models import load_model
import pickle

print("Please type model foldername : ")
mn = input()
mn = str(mn)
print("[INFO] loading network...")
lb = pickle.loads(open(os.path.join("eval/",mn,"BTv512lb"), "rb").read())
model = load_model(os.path.join("eval/",mn,"my_best_model.hdf5"))

Tk().withdraw()
path2 = r"" + askopenfilename()
print(path2)
sample = cv2.imread(path2, 0)
edges = img_proc.findcontr(sample)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total_le = 0
total_wi = 0
total_ar2 = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    total_ar2 += area
    length, width,_ = img_proc.findlength(sample,cnt, 'none')
    total_le += float(length)
    total_wi += float(width)
avgtotal = total_ar2 / len(contours)
avgAR = total_le / total_wi
print("Avg Length :",total_le)
hl = float(total_le / len(contours)) * 0.75
print("Large broken :", hl, "mm")
lm = float(total_le / len(contours)) * 0.25
print("Small broken :", lm, "mm")

Tk().withdraw()
path = r"" + askopenfilename()
print(path)
img = cv2.imread(path, 0)
img2 = cv2.imread(path)
img3 = cv2.imread(path)
edges2 = img_proc.findcontr(img)
cons, hierarchy = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

total_arlb=0
total_arsb=0
total_arh=0
count=0
total=len(cons)
pbar = tqdm(total)
for cnt in cons:
    x, y, w, h = cv2.boundingRect(cnt)
    x, y, w, h = x - 10, y - 10, w + 20, h + 20
    roi = img2[y:y + h, x:x + w]
    roi2 = img3[y:y + h, x:x + w]
    area = cv2.contourArea(cnt)
    #box = img_proc.minreact(cnt, img3)
    length, width,roi = img_proc.findlength(img3,cnt,roi)
    id = img_proc.filter(length, hl, lm, area, width)
    if (id == "Double"):
        pass

    if (id == "SB"):
        area= cv2.contourArea(cnt)
        total_arsb+=area
        count+=1
        output = roi
        minEllipse= img_proc.ovalmasking(cnt)
        color = (0,0,0)
        test = img_proc.smallbroken(img3, id, length, x, y, w, h)
        #cv2.ellipse(img2, minEllipse, color, -1)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 0), -1)

    if (id == "LB"):
        area= cv2.contourArea(cnt)
        total_arlb+=area
        count += 1
        output = roi
        minEllipse= img_proc.ovalmasking(cnt)
        color = (0,0,0)
        test = img_proc.smallbroken(img3, id, length, x, y, w, h)
        #cv2.ellipse(img2, minEllipse, color, -1)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 0), -1)

    if (id == "Headrice"):
        area= cv2.contourArea(cnt)
        total_arh+=area
        count += 1
        print(count)
        print(length)
        print(width)
        shape = length / width
        l = ""
        if (shape >= 3.0):
            l = "slender"
        elif (shape < 3.0 and shape >= 2.1):
            l = "medium"
        elif (shape < 2.1 and shape >= 1.1):
            l = "Bold"
        elif (shape <= 1.1):
            l = "Round"
        AR = ""
        if (shape < 2.05):
            output = roi
            label = "OV1(AP)"
        else:
            roi = img_proc.imagerotation(cnt, roi)
            #output = roi
            label = img_proc.resizing(roi, model,lb)
            #img_proc.imgcrop(roi, label)
        img_proc.goodrice(img3, label, width, length, x, y, w, h)
        x, y, w, h = cv2.boundingRect(cnt)
        #img_proc.imgcrop(output, label)
        #img_proc.ovalimgcrop(img2, label, cnt)
        minEllipse= img_proc.ovalmasking(cnt)
        color = (0,0,0)
        #cv2.ellipse(img2, minEllipse, color, -1)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 0), -1)

    pbar.update(1)
pbar.close()
#cv2.imwrite('BT' + '.png', img2)
#cv2.imwrite('BT2' + '.png', img3)
#print(" cnt:",total_arh,total_arlb,total_arsb,count)
if (count < total):
    cv2.imwrite('BT2' + '.png', img3)
    img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    edges2 = img_proc.findcontr(img)
    cons, hierarchy = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pbar = tqdm(total=len(cons))
    for cnt in cons:
        x, y, w, h = cv2.boundingRect(cnt)
        x, y, w, h = x - 10, y - 10, w + 20, h + 20
        roi = img2[y:y + h, x:x + w]
        roi2 = img3[y:y + h, x:x + w]
        area = cv2.contourArea(cnt)
        length, width, roi2 = img_proc.findlength(img3,cnt, roi2)
        id = img_proc.filter(length, hl, lm, area, width)
        if (id == "Double"):
            image = roi
            image2 = img3[y:y + h, x:x + w]
            labels = img_proc.watersheding(image)
            for label in np.unique(labels):
                if label == 0:
                    continue
                c = img_proc.labeling(labels, label, image)
                area2 = cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)
                minEllipse = img_proc.ovalmasking(c)
                color = (0, 0, 0)
                koi = image[y:y + h, x:x + w]
                le, wi, koi = img_proc.findlength(c, koi)
                sid = img_proc.filter(le, hl, lm, area2, wi)
                if (sid == "LB"):
                    count += 1
                    area = cv2.contourArea(c)
                    total_arlb += area
                    img_proc.smallbroken(image2, sid, le, x, y, w, h)
                    cv2.ellipse(image, minEllipse, color, -1)
                    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
                if (sid == "SB"):
                    count += 1
                    area = cv2.contourArea(c)
                    total_arsb += area
                    img_proc.smallbroken(image2, sid, le, x, y, w, h)
                    # cv2.ellipse(image, minEllipse, color, -1)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
                else:
                    # cv2.imshow("before",koi)
                    area = cv2.contourArea(c)
                    total_arh += area
                    count += 1
                    koi = img_proc.imagerotation(c, koi)
                    # cv2.imshow("After", koi)
                    # cv2.waitKey(0)
                    label = img_proc.resizing(koi, model, lb)
                    # img_proc.ovalimgcrop(image, label, c)
                    img_proc.goodrice(image2, label, wi, le, x, y, w, h)
                    # cv2.ellipse(image, minEllipse, color, -1)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
                    """
                    if label == "DUO":
                        localMax = peak_local_max(D, indices=False, min_distance=20, labels=binary)
                        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
                        labels = watershed(-D, markers, mask=binary)
                        for label in np.unique(labels):
                            if label == 0:
                                continue
                            mask = np.zeros(img.shape, dtype="uint8")
                            mask[labels == label] = 255
                            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cnts = imutils.grab_contours(cnts)
                            c = max(cnts, key=cv2.contourArea)
                            area2 = cv2.contourArea(c)
                            x, y, w, h = cv2.boundingRect(c)
                            koi = image[y:y + h, x:x + w]
                            label = img_proc.resizing(koi)
                            img_proc.imgcrop(koi, label)
                            img_proc.goodrice(image2, label, wi, le, x, y, w, h)
                            #color = (0,0,0)
                            #cv2.ellipse(koi, minEllipse, color, -1)
                            #cv2.rectangle(koi, (x, y), (x + w, y + h), (0, 0, 0), -1)
                    else:
                        #minEllipse = img_proc.ovalmasking(koi)
                        #koi = img_proc.masking(koi, minEllipse)
                        label = img_proc.resizing(koi)
                        img_proc.imgcrop(koi, label)
                        img_proc.goodrice(image2, label, wi, le, x, y, w, h)
                        #color = (0, 0, 0)
                        #cv2.ellipse(koi, minEllipse, color, -1)
                        #cv2.rectangle(koi, (x, y), (x + w, y + h), (0, 0, 0), -1)
                    """
        pbar.update(1)
    pbar.close()

print("Plused :",total_arh,total_arlb,total_arsb,count)
img_proc.finalweight(total_arh,total_arlb,total_arsb,count,img3)
cv2.imwrite('BT2' + '.png', img3)
#cv2.imwrite('BT' + '.png', img2)