import numpy as np
import matplotlib.pyplot as plt

def convert_to_image (data, shape = None):
	import numpy as np		
	normalized = (data-np.min(data))/(np.max(data)-np.min(data))
	data = normalized * 255
	
	from scipy.misc import toimage
	im = toimage(data)
	if shape != None:
		im = im.resize(shape)
	return im
			
def spectrum(signal, taper = True):
	windowed = signal
	if taper:
		windowed = windowed * np.blackman(len(windowed))
	a = abs(np.fft.rfft(windowed))

	#db = 20 * np.log10(a)

	return a

def spectrogram (data):
	spec_matrix = []
	db_matrix = []
	for i in range(len(data)):
		a = spectrum(data[i], False)
		spec_matrix.append (a)
		db = 20 * np.log10(a)		 
		db_matrix.append (db)

	spec_matrix = np.array(spec_matrix)
	db_matrix = db_matrix - np.amax(db_matrix)
	db_matrix = np.nan_to_num(db_matrix)
	db_matrix[db_matrix == -np.inf] = 0
	return db_matrix

def fk (data):
	#data = data*np.blackman(len(data[0]))
	data = data.T
	freq = np.fft.fft2(data)
	freq = np.fft.fftshift(freq)
	freq = freq[int(len(freq)/2):,:]

	#print (np.fft.rfftfreq(freq.shape[0], self.dt_synt))
	#print (np.fft.fftfreq(freq.shape[1], self.dx_synt))

	freq = np.abs(freq)
	freq = 20 * np.log10(freq)
	freq = freq - np.amax(freq)
	freq = np.nan_to_num(freq)
	freq[freq == -np.inf] = 0
	return freq

def plot(data):
	fig = plt.figure(figsize=(16, 8))
	ax = fig.add_subplot(111)

	# How to remove decorations from matplotlib: https://stackoverflow.com/questions/38411226/matplotlib-get-clean-plot-remove-all-decorations
	#ax.set_axis_off()

	data = data.T
	vm = np.percentile(data, 99)
	imparams = {
		#'interpolation': 'none',
		'cmap': "gray",
		'vmin': -vm,
		'vmax': vm,
		'aspect': 'auto'
	}
	plt.imshow(data, **imparams)
	#plt.colorbar()
	#plt.show()
	return plt

def plot_spec(data):
	fig = plt.figure(figsize=(16, 8))
	ax = fig.add_subplot(111)

	# How to remove decorations from matplotlib: https://stackoverflow.com/questions/38411226/matplotlib-get-clean-plot-remove-all-decorations
	#ax.set_axis_off()

	imparams = {
		#'interpolation': 'none',
		'cmap': "gray",
		'aspect': 'auto'
	}
	plt.imshow(data, **imparams)
	#plt.colorbar()
	#plt.show()
	return plt

		
class FreqNoiser:
	def __init__(self):
		pass
		
	def run (self, handler):
		handler.data = self.addNoiseStatic(handler.data)
		
	@staticmethod 
	def addNoiseStatic (data):
		import numpy as np
		data_new = []
		for t in data:
			f = np.fft.rfft(t)
			l = len (f)
			llow = int(l*0.5);
			lhi = int(l*0.75)
			for i in range (l):
				if i < llow:
					sc = 0
				if i in range(llow, lhi):
					sc = (i - llow)/(lhi - llow)
				if i > lhi:
					sc = 1
				
				f [i] *= 1 + sc*10
			
			t_new = np.fft.irfft(f)
			data_new.append (t_new)
		return np.array(data_new)

		
class BP:
	def __init__(self, f1, f2, f3, f4, factor):
		self.f1 = f1
		self.f2 = f2
		self.f3 = f3
		self.f4 = f4
		self.factor = factor
		pass
		
	def run (self, handler):
		handler.data = self.bp(handler.data)
		
	def bp (self, data):
		import numpy as np
		data_new = []
		f = np.fft.rfft(data[0])
		l = len (f)
		f1 = int(l*self.f1)
		f2 = int(l*self.f2)
		f3 = int(l*self.f3)
		f4 = int(l*self.f4)
		for t in data:
			f = np.fft.rfft(t)
			for i in range (l):
				
				sc = 0
				if i in range(f1, f2):
					sc = (i - f1)/(f2 - f1)
				if i in range(f2, f3):
					sc = 1
				if i in range(f3, f4):
					sc = 1 - (i - f3)/(f4 - f3)
					
				f [i] *= 1 + sc*self.factor
			
			t_new = np.fft.irfft(f)
			data_new.append (t_new)
		return np.array(data_new)
		
		
def create_images (data, png_name = None):
	from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
	
	image = convert_to_image (data.T)
	im_data = img_to_array (image)

	fk_data = fk (data)
	fk_image = convert_to_image (fk_data)
	im_fk_data = img_to_array (fk_image)
	
	spec_data = spectrogram (data)
	spec_image = convert_to_image (spec_data.T)
	im_spec_data = img_to_array (spec_image)
	
	if png_name != None:
		image.save (png_name + '_data.png')
		fk_image.save (png_name + '_fk.png')
		spec_image.save (png_name + '_spec.png')
	
	return [im_data, im_fk_data, im_spec_data]
	
def create_dirs (images_dir, num_classes):
	import os
	if os.path.exists(images_dir):
		import shutil
		shutil.rmtree(images_dir)

	if not os.path.exists(images_dir):
		os.makedirs(images_dir)

	train_data_dir = images_dir + 'train/'
	test_data_dir = images_dir + 'test/'

	if not os.path.exists(train_data_dir):
		os.makedirs(train_data_dir)

	if not os.path.exists(test_data_dir):
		os.makedirs(test_data_dir)
		
	for i in range(num_classes):
		a = train_data_dir + str(i)
		if not os.path.exists(a):
			os.makedirs(a)
			
		a = test_data_dir + str(i)
		if not os.path.exists(a):
			os.makedirs(a)
		
	return train_data_dir, test_data_dir
				
def create_seis_dataset (file_name, sorting_key, window, noisers, values, images_dir = 'outputs/images/'):
	import numpy as np
	import seismic_handler

		
	X_train = [[],[],[]]
	y_train = []
	X_test = [[],[],[]] 
	y_test = []
	v_train = []
	v_test = []
	  

	import os
	if os.path.exists('outputs/X_train0.npy'):
		for i in range(len(X_train)): 
			X_train[i] = np.load('outputs/X_train' + str(i) + '.npy')
			X_test[i] = np.load('outputs/X_test' + str(i) + '.npy')
		
		y_train = np.load('outputs/y_train.npy')
		y_test = np.load('outputs/y_test.npy')

		v_train = np.load('outputs/v_train.npy')
		v_test = np.load('outputs/v_test.npy')
		
		return (X_train, y_train, v_train), (X_test, y_test, v_test)
	
	if images_dir != None:
		train_data_dir, test_data_dir = create_dirs(images_dir, len(noisers))
			
	handlers = [seismic_handler.SeismicPrestack(file_name, noiser) for noiser in noisers]
	gather_keys = [handler.getHeaderVals (sorting_key) for handler in handlers]
	
	counter = 0
	nhandlers = len (handlers)
		   
	for key in gather_keys[0]:
		partss = [handler.readGatherParts (key, window[0], window[1]) for handler in handlers]		 
		sh = partss[0][0].shape
		
		for category in range(nhandlers):
			nparts = len (partss[category])
			for i in range(nparts):
				
				# just sparser of data
				if category != i % nhandlers:
					continue

				data = partss[category][i]
				if data.shape != sh:
					throw ('wrong shapes')
				
				train = False
				if np.random.rand() < 0.8:
					train = True
					
				png_name = None
				if images_dir != None:
					if train:
						png_name = train_data_dir
					else:
						png_name = test_data_dir
			   
					png_name += str (category) + '/'	
					png_name += str(counter)
								   
				data_repr = create_images (data, png_name)				 

				if train:
					for i in range (len(data_repr)):
						X_train[i].append(data_repr[i])
					y_train.append (category)
					v_train.append(values[category])
				else:
					for i in range (len(data_repr)):
						X_test[i].append(data_repr[i])
					y_test.append(category)	 
					v_test.append(values[category])
				
				counter += 1
	
	y_test = np.array(y_test)
	y_train = np.array(y_train)
	for i in range (len(X_train)):
		X_train[i] = np.array(X_train[i])
		X_test[i] = np.array(X_test[i])
		if len(X_train[i]) != len(X_train[0]):
			print ('len(X_train[i]) != len(X_train[0])')
			return
		if len(X_test[i]) != len(X_test[0]):
			print ('len(X_test[i]) != len(X_test[0])')
			return
	
	if len(X_train[0]) != len(y_train):
		print ('len(X_train) != len(y_train)')
		return
	if len(X_test[0]) != len(y_test):
		print ('len(X_test) != len(y_test)')
		return
		
	for i in range(len(X_train)): 
		np.save('outputs/X_train' + str(i), X_train[i])
		np.save('outputs/X_test' + str(i), X_test[i])
	
	np.save('outputs/y_train', y_train)
	np.save('outputs/y_test', y_test)
	np.save('outputs/v_train', v_train)
	np.save('outputs/v_test', v_test)
	return (X_train, y_train, v_train), (X_test, y_test, v_test)