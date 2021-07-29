import os, sys
import pdb
import random
import h5py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras import backend as K

# Set seed just to have reproducible results
np.random.seed(84)
tf.random.set_seed(84)
random.seed(84)

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

def deepNN_model(input_shape):
	model = Sequential()
	model.add(Dense(32, input_shape=np.expand_dims(input_shape, axis=0), activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(64, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(256, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(1, activation='linear'))	
	return model

def NN_model(input_shape):
	model = Sequential()
	model.add(Dense(32, input_shape=np.expand_dims(input_shape, axis=0), activation='relu'))
	model.add(Dense(64))
	model.add(BatchNormalization())
	model.add(Dense(64, activation='relu'))
	model.add(Dense(128))
	model.add(BatchNormalization())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(1))	
	return model


def load_model(model_path):
	dependencies = {'soft_acc': soft_acc}
	model = tf.keras.models.load_model(model_path, custom_objects=dependencies)
	return model


def scaling(arr, max_int, type='minmax'):
	if type == 'minmax':
		out = arr/max_int
	return out


def gallonsOfWater(rnd_arr):

	N = len(rnd_arr)

	left_arr = np.zeros(N).astype(int)
	right_arr = np.zeros(N).astype(int)

	for i in np.arange(1,N-1):

		if rnd_arr[i] < left_arr[i-1]:
			left_arr[i] = np.max([rnd_arr[i], left_arr[i-1]])
		else:
			if i == 1:
				left_arr[i] = np.max([rnd_arr[i], rnd_arr[i-1]])
			else:
				left_arr[i] = np.max([left_arr[i-1], rnd_arr[i]])

		if rnd_arr[N-i-1] < right_arr[N-i]:
			right_arr[N-i-1] = np.max([rnd_arr[N-i-1], right_arr[N-i]])
		else:
			if i == 1:
				right_arr[N-i-1] = np.max([rnd_arr[N-i-1], rnd_arr[N-i]])
			else:
				right_arr[N-i-1] = np.max([rnd_arr[N-i-1], right_arr[N-i]])

	out = np.min([left_arr, right_arr], axis=0) - rnd_arr
	out = np.sum(out[1:-1])

	return out


def arrayGenerator(batch_size, max_len, max_int):

	index = 0
	
	batch_feats = np.zeros((batch_size, max_len), dtype=np.float32)
	batch_labels = np.zeros((batch_size, 1), dtype=np.float32)

	while True:
		
		rnd_len = random.randint(0,max_len)
		rnd_arr = np.random.rand(rnd_len)
		rnd_arr = np.round(rnd_arr*max_int).astype(int)

		y_true = gallonsOfWater(rnd_arr)
		y_true = y_true / 800

		rnd_arr = np.hstack((rnd_arr, np.zeros(max_len - len(rnd_arr))))

		rnd_arr = scaling(rnd_arr, max_int, type='minmax')

		if index == batch_size-1:
			batch_feats[index] = rnd_arr
			batch_labels[index] = y_true
			index = 0
			yield (batch_feats , batch_labels)
		else:
			batch_feats[index] = rnd_arr
			batch_labels[index] = y_true
			index += 1


def main():

	batch_size = 16
	max_len = 10
	max_int = 10
	lr = 0.001
	nepochs = 10
	optimizer = Adam(lr=lr)
	tr_step_per_epoch = 1000
	loss = 'mae'
	metrics = 'mse'
	trainable_model = ''
	out_model_name = 'deepNN'
	mode = 'train'
	N_test_samples = 1000


	arr_gen = arrayGenerator(batch_size, max_len, max_int)

	#NN model
	model = deepNN_model(max_len)
	# model = NN_model(max_len)

	if trainable_model:
		model_path = os.getcwd() + '/output/'+trainable_model+'/'+trainable_model+'.h5'
		model = load_model(model_path)

	if mode == 'train':

		print(model.summary())

		model.compile(loss=loss, optimizer=optimizer,
                metrics=[metrics])

		#Fit model
		model.fit_generator(arr_gen, steps_per_epoch=tr_step_per_epoch, epochs=nepochs, verbose=1)

		odir = os.getcwd() + '/output/' + out_model_name

		if not os.path.exists(odir):
			os.makedirs(odir)

		model.save(os.path.join(odir, out_model_name+'.h5'), overwrite=True)

	elif mode == 'test':

		test_arr_gen = arrayGenerator(N_test_samples, max_len, max_int)

		X, y = next(test_arr_gen)
		y = np.round(y*800)

		y_pred = model.predict(X)
		y_pred = np.round(y_pred*800)

		correct = (y == y_pred)
		accuracy = correct.sum() / correct.size

		print("Accuracy: " + str(accuracy))


	print('>>> DONE <<<')
	
        	


if __name__ == '__main__':
	main()