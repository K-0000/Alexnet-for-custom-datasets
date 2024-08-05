import tensorflow as tf
import numpy as np
import cv2
from imutils import paths
import os
import pickle
#from keras.utils import img_to_array
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
#from keras.optimizers import adam_v2
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from keras.layers import BatchNormalization
#from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation,Dropout,Flatten,Dense
from keras.models import Sequential
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib
from sklearn.model_selection import KFold, StratifiedKFold
import numpy
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

print("Please type Batch Size : ")
bn = input()
bn = int(bn)
BS = bn
IMAGE_DIMS = (224,224,3)
print("Please type model foldername : ")
mn = input()
mn = str(mn)
print("[INFO] loading network...")
lb = pickle.loads(open(os.path.join("eval/",mn,"BTv512lb"), "rb").read())
model = tf.keras.models.load_model(os.path.join("eval/",mn,"my_best_model.hdf5"))

print("Please type model foldername : ")
fn = input()
fn = str(fn)
#print(os.path.join(fn+"/testing"))
print ("[INFO] loading test images...")
testPaths = sorted(list(paths.list_images(os.path.join(fn+"/testing"))))
#testPaths = sorted(list(paths.list_images("SWT/testing")))



test =[]
y=[]
try:
	for imagePath in testPaths:
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
		image = img_to_array(image)
		test.append(image)
		label = imagePath.split(os.path.sep)[-2]
		y.append(label)

	X = np.array(test, dtype="float")/255.0
	Y = np.array(y)
	lb = LabelBinarizer()
	Y  = lb.fit_transform(Y)
	print("[INFO] Calculating model accuracy")
	scores = model.evaluate(X,Y)
	print(f"Test Accuracy: {scores[1]*100}")

	print("[INFO] Evaluating Network...")
	preIdxs = model.predict(X, batch_size=BS)
	preIdxs = np.argmax(preIdxs, axis=1)

	print(classification_report(Y.argmax(axis=1), preIdxs,target_names=lb.classes_))

	cm = confusion_matrix(Y.argmax(axis=1),preIdxs)
	total= sum(sum(cm))
	acc = (cm[0,0]+cm[1,1])/total
	sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
	specificity = cm[1,1]/(cm[1,0]+cm[1,1])

	print(cm)
	print("acc:{:.4f}".format(acc))
	print("sensitivityacc:{:.4f}".format(sensitivity))
	print("specificity :{:.4f}".format(specificity ))

	input("Press Enter to continue...")

except Exception as e:
	print(e)
	input("Press Enter to continue...")