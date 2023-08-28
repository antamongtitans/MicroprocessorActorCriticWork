from keras import applications
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D, LeakyReLU, PReLU
from keras.callbacks import CSVLogger, ModelCheckpoint

#try new LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

#Stuff needed for dataset
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import operator
import math

#needed for storage
import pickle
import sys
sys.setrecursionlimit(10000)

from dataStructures import *

chip_array = pickle.load( open( "save_chips_1000.p", "rb" ) )

#make sure we have data
#Uncomment to print
initial_print = False
if (initial_print):
	for item in chip_array:
		item.print_plot()


	num_rectangles = 0
	#Get average number of rectangles
	for item in chip_array:
		rect_list = item.get_rectangle_list()
		tmp_rect_len = len(rect_list)
		num_rectangles = num_rectangles + tmp_rect_len

	average_num_rect = num_rectangles / len(chip_array)
	plt.show()

#calculate value for test and training
#percentage for testing
perc_test = 0.8
total_entries = 0
test_num = int(len(chip_array) * perc_test)
#Setup Data with inputs as data and outputs as labels
#First level is chip
#Second level of array name, width, height, connection, 
#	bounds_xmin, bounds_ymin, bounse_xmax, bounds_ymax
#second level for label is x cord, y cord
train_datas = []
train_labels = []
for chip in chip_array[:test_num]:
	#get rectangles
	rect_list = chip.get_rectangle_list()
	for rectangle in rect_list:
		#get connections for each rectangle
		tmp_connection_list = rectangle.get_connect()
		for connection in tmp_connection_list:
			train_datas.append([rectangle.get_name(), rectangle.get_width(),
				rectangle.get_height(), connection.get_name(), 0, 0, 
				chip.get_width(), chip.get_height()])
			train_labels.append([rectangle.get_minx(), rectangle.get_miny()])

verf_datas = []
verf_labels = []
for chip in chip_array[test_num:]:
	#get rectangles
	rect_list = chip.get_rectangle_list()
	for rectangle in rect_list:
		#get connections for each rectangle
		tmp_connection_list = rectangle.get_connect()
		for connection in tmp_connection_list:
			verf_datas.append([rectangle.get_name(), rectangle.get_width(),
				rectangle.get_height(), connection.get_name(), 0, 0,
				chip.get_width(), chip.get_height()])
			verf_labels.append([rectangle.get_minx(), rectangle.get_miny()])


np_train_datas = np.array(train_datas)
np_train_labels = np.array(train_labels)
np_verf_datas = np.array(verf_datas)
np_verf_labels = np.array(verf_labels)
#shape of data
#since variable size step_size is none, features is width of data
step_size = 8
nb_features = len(train_datas[0])
batch_size = 128
epochs = 200

print(np_train_datas.shape[0])
print(np_verf_datas.shape[0])


#For LSTM
np_train_datas = np.reshape(np_train_datas, 
	(np_train_datas.shape[0], 1, np_train_datas.shape[1]))
np_train_labels = np.reshape(np_train_labels, 
	(np_train_labels.shape[0], 1, np_train_labels.shape[1]))
np_verf_datas = np.reshape(np_verf_datas, 
	(np_verf_datas.shape[0], 1, np_verf_datas.shape[1]))
np_verf_labels = np.reshape(np_verf_labels,
	(np_verf_labels.shape[0], 1, np_verf_labels.shape[1]))

model = Sequential()

#model.add(Embedding(max_features, output_dim=256)) Shouldn't need this since its all standardized before

model.add(LSTM(8, input_shape=(1,8), return_sequences=True))
model.add(Dropout(0.5))
#2 is number of output layers for Dense
model.add(Dense(2, activation='sigmoid'))

model.summary()
model.compile(loss='mse',
              optimizer='nadam',
              metrics=['accuracy'])

model.fit(np_train_datas, np_train_labels, batch_size=1000, epochs=10)
score = model.evaluate(np_verf_datas, np_verf_labels, batch_size=128)

print(score)
