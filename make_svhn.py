import numpy as np
import scipy.io
from array import array

read_input = scipy.io.loadmat('test_32x32.mat')
j=0
output_file = open('test_batch_%d.bin' % j, 'ab')

for i in range(0, 64000):
		# read to bin file
		if	i>0 and i % 12800 == 0:
			output_file.close()
			j=j+1
			output_file = open('data_batch_%d.bin' % j, 'ab')

		# Write to bin file
		if	read_input['y'][i] == 10:
			read_input['y'][i] = 0
			read_input['y'][i].astype('uint8').tofile(output_file)
			read_input['X'][:,:,:,i].astype('uint32').tofile(output_file)
output_file.close()
