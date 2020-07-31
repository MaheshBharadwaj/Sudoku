import numpy as np
import pandas as pd
import cv2
import os

import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

from keras import Sequential
from keras.layers import *
from keras.regularizers import *
from keras.callbacks import *
from keras.optimizers import Adam

import source.utils as utils


def main():
	stream = os.popen('git clone https://github.com/wichtounet/sudoku_dataset/')


	df = pd.read_csv('./sudoku_dataset/outlines_sorted.csv')

	# Strip extension from path
	df['filepath'] = df['filepath'].apply(
		lambda path: './sudoku_dataset' + path[1:-4])


	X = []
	y = []

	for i in tqdm(range(df.shape[0])):
		path = str(df.iloc[i, 0])
		X.append(utils.process_image(path + '.jpg'))
		y.append(utils.ground_truth(path + '.dat'))


	X = np.array(X)
	y = np.array(y)


	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42)

	print('Train Shape: ', X_train.shape)
	print('Test  Shape: ', X_test.shape)

	print('Train Shape: ', X_train.shape)
	print('Test  Shape: ', X_test.shape)


	# Training digit classifier
	digit_train_img = []
	digit_train_truth = []
	for i, sudoku in enumerate(X_train):
		digit_train_img.extend(utils.get_cells(sudoku, size=288))
		digit_train_truth.extend(y_train[i])

	digit_train_truth = np.array(digit_train_truth)
	digit_train_img = np.array(digit_train_img)


	digit_train_img = np.reshape(
		digit_train_img, (digit_train_img.shape[0], 32, 32, 1))
	digit_train_truth = to_categorical(digit_train_truth, 10)

	d_train, d_test, v_train, v_test = train_test_split(
		digit_train_img, digit_train_truth, test_size=0.2)

	print('Train digits: ', d_train.shape)
	print('Test  digits: ', d_test.shape)

	print('Train values: ', v_train.shape)
	print('Test  values: ', v_test.shape)




	
	if os.path.exists('./model.h5'):
		model = keras.models.load_model('./model.h5')

	else:
		model = Sequential()
		model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 1), padding='same',
				activation='relu', kernel_regularizer=l2()))
		model.add(Conv2D(32, (3, 3), padding='same',
				activation='relu', kernel_regularizer=l2()))
		model.add(MaxPooling2D((2, 2)))
		model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2)))
		model.add(Flatten())
		model.add(Dropout(0.3))
		model.add(Dense(10, activation='softmax'))
		model.summary()
		model.compile(metrics=['accuracy'], loss='categorical_crossentropy',
					optimizer=Adam(learning_rate=5e-4))


		LR_reduce = ReduceLROnPlateau(
						monitor='val_accuracy',
						factor=.67,
						patience=10,
						min_lr=.5e-6,
						verbose=1)

		ES_monitor = EarlyStopping(monitor='val_loss', patience=15)
		history = model.fit(
					d_train,
					v_train,
					batch_size=256,
					validation_data=(d_test, v_test),
					epochs=300,
					callbacks=[LR_reduce, ES_monitor, utils.myCallback()])
		model.save('model.h5')
		plt.plot(history.history['accuracy'])
		plt.plot(history.history['val_accuracy'])
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.title('Accuracy vs Epoch')
		plt.legend(['Training', 'Validation'], loc='lower right')
		plt.savefig('Acc.png')

		plt.figure()
		plt.plot(history.history['loss'], label='Training data')
		plt.plot(history.history['val_loss'], label='Validation data')
		plt.title('Loss')
		plt.ylabel('Loss value')
		plt.title('Loss vs Epoch')
		plt.xlabel('No. epoch')
		plt.legend(loc="upper left")
		plt.savefig('Loss.png')
		

	v_pred = model.predict(d_test)
	v_pred = np.argmax(v_pred, axis=1)
	v_test_rep = np.argmax(v_test, axis=1)

	print(classification_report(
  		y_true=v_test_rep,
  		y_pred=v_pred
		))


	#Testing on unseen sudokus

	accurate = 0
	total = 0

	for i in range(len(X_test)):
		sudoku = X_test[i]
		cells = utils.get_cells(sudoku, size=288)
		cells = np.reshape(cells, (cells.shape[0], 32, 32, 1))
		v_true = y_test[i]
		v_pred = model.predict(cells)
		v_pred = np.argmax(v_pred, axis=1)
		accurate += utils.compare_outputs(pred=v_pred, truth=v_true)
		total +=1

	print('Accuracy: ', (accurate/total))

	utils.get_sudoku('./test1.jpeg', model)
	utils.get_sudoku('./test2.jpg', model)
	utils.get_sudoku('./test3.jpeg', model)
if __name__ == '__main__':
	main()