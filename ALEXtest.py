import tensorflow as tf
import numpy as np
import cv2
from imutils import paths
import random
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

seed = 7
numpy.random.seed(seed)
EPOCHS = 10
INIT_LR = 1e-5
BS = 64
IMAGE_DIMS = (224,224,3)
data  = []
labels =[]

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

print("Please type training foldername : ")
fn = input()
fn = str(fn)
print ("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(fn+"/training")))
np.random.seed(100),random.shuffle(imagePaths)

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	if image is None:
		print(f"[WARNING] Unable to load image at path: {imagePath}")
		continue
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	#print("pass")
	image = img_to_array(image)
	data.append(image)
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

data = np.array(data, dtype="float")/255.0
labels = np.array(labels)
print(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
(trainX, testX, trainY, testY)= train_test_split(data, labels, test_size=0.10, random_state=42)
aug = ImageDataGenerator(rotation_range=90, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05, zoom_range=0.05, horizontal_flip=True, vertical_flip=True, fill_mode="nearest")
#aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05, zoom_range=0.05, horizontal_flip=True, fill_mode="nearest")
try:
	model = Sequential()
	inputShape = (224, 224, 3)
	chanDim = -1
	if K.image_data_format() == "channels_first":
		inputShape = (224, 224, 3)
		chanDim = 1
	model.add(Conv2D(96, (11, 11), padding="valid", input_shape=inputShape, strides=(4, 4)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
	model.add(Conv2D(256, (5, 5), padding="valid", input_shape=inputShape, strides=(1, 1)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
	model.add(Conv2D(384, (3, 3), padding="valid", input_shape=inputShape, strides=(1, 1)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(384, (3, 3), padding="valid", input_shape=inputShape, strides=(1, 1)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(256, (3, 3), padding="valid", input_shape=inputShape, strides=(1, 1)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
	model.add(Flatten())
	model.add(Dense(4096, input_shape=inputShape))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Dropout(0.4))


	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dropout(0.4))
	model.add(Dense(1000))
	model.add(Activation('relu'))
	#model.add(Dropout(0.4))
	#model.add(Dense(512))
	#model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))
	model.add(Dense(1))
	#model.add(Activation('softmax'))
	model.add(Activation('sigmoid'))
	print(model.summary())

	print("[INFO] serializing label binarizer...")
	f = open("eval/BTv512lb", "wb")
	f.write(pickle.dumps(lb))
	f.close()
	filepath = 'eval/my_best_model.hdf5'
	checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
	callbacks = [checkpoint]


	opt = Adam(learning_rate=INIT_LR, decay=INIT_LR/ EPOCHS)
	#model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
	model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
	print ("[INFO] training network......")
	H = model.fit(
    	aug.flow(trainX, trainY, batch_size=BS),
    	validation_data=(testX, testY),
    	steps_per_epoch=len(trainX) // BS,
    	epochs=EPOCHS, verbose=1, callbacks=callbacks)


	# save the model to disk
	print("[INFO] serializing network...")
	model.save("BTv512.h5")
	# save the label binarizer to disk
	print("[INFO] serializing label binarizer...")
	f = open("BTv512lb", "wb")
	f.write(pickle.dumps(lb))
	f.close()

	# plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	N = EPOCHS
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="upper left")
	plt.savefig("new75.png")

except Exception as e:
	print(e)
	pass

print ("[INFO] loading test images...")
testPaths = sorted(list(paths.list_images(fn+"/testing")))
np.random.seed(100),random.shuffle(imagePaths)

test =[]
y=[]
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
preIdxs = np.argmax(preIdxs, axis=2)

print(classification_report(Y.argmax(axis=2), preIdxs,target_names=lb.classes_))

cm = confusion_matrix(Y.argmax(axis=2),preIdxs)
total= sum(sum(cm))
acc = (cm[0,0]+cm[1,1])/total
sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
specificity = cm[1,1]/(cm[1,0]+cm[1,1])

print(cm)
print("acc:{:.4f}".format(acc))
print("sensitivityacc:{:.4f}".format(sensitivity))
print("specificity :{:.4f}".format(specificity ))



"""
X = np.array(data, dtype="float")/255.0
Y = np.array(labels)

lb=LabelBinarizer()
le=LabelEncoder()
print ("[info] data matrix: {:.2f}MB".format(X.nbytes/(1024*1000.0)))
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05, zoom_range=0.05, horizontal_flip=True, fill_mode="nearest")

for train, test in kfold.split(X,Y):
	aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05,
							 zoom_range=0.05, horizontal_flip=True, fill_mode="nearest")

	#print("TRAIN:",train,"TEST:",test)
	#print("YTEST:",X[test],"YTRAIN:",Y[test])
	model = Sequential()
	inputShape = (224, 224, 3)
	chanDim = -1
	if K.image_data_format() == "channels_first":
		inputShape = (224, 224, 3)
		chanDim = 1
	model.add(Conv2D(96, (11, 11), padding="valid", input_shape=inputShape, strides=(4, 4)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
	model.add(Conv2D(256, (5, 5), padding="valid", input_shape=inputShape, strides=(1, 1)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
	model.add(Conv2D(384, (3, 3), padding="valid", input_shape=inputShape, strides=(1, 1)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(384, (3, 3), padding="valid", input_shape=inputShape, strides=(1, 1)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(256, (3, 3), padding="valid", input_shape=inputShape, strides=(1, 1)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
	model.add(Flatten())
	model.add(Dense(4096, input_shape=inputShape))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Dropout(0.4))
	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dropout(0.4))
	model.add(Dense(1000))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))
	model.add(Dense(8))
	model.add(Activation('softmax'))
	trainX, testX = X[train], X[test]
	trainY, testY = lb.fit_transform(Y[train]), lb.fit_transform(Y[test])
"""
'''	
#print(model.summary())
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss='CategoricalCrossentropy', optimizer=opt, metrics=['accuracy'])
	model.fit(aug.flow(X[train], lb.fit_transform(Y[train]),batch_size=BS),validation_data=(X[test],lb.fit_transform(Y[test])), epochs=EPOCHS, verbose=1)
		# evaluate the model
	scores = model.evaluate(X[test], lb.fit_transform(Y[test]), verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
	cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
'''