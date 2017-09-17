#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os, sys, sklearn, scipy, librosa


from optparse import OptionParser
from matplotlib.pyplot import specgram
from scipy.io import wavfile
from scipy.stats import skew, kurtosis

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


label_mapping = { 'BassClarinet':0, 'BassTrombone':1, 'BbClarinet':2, 'Cello':3, 'EbClarinet':4, 'Marimba':5, 'TenorTrombone':6, 'Viola':7, 'Violin':8, 'Xylophone':9 }
rlabel_mapping = { 0:'BassClarinet', 1:'BassTrombone', 2:'BbClarinet', 3:'Cello', 4:'EbClarinet', 5:'Marimba', 6:'TenorTrombone', 7:'Viola', 8:'Violin', 9:'Xylophone' }

def calculate_stats( data, a ):

	mean = np.mean( data, axis=a )
	var = np.var( data, axis=a )
	skewness = skew( data, axis=a )
	kurt = kurtosis( data, axis=a )
	p5 = np.percentile( data, 5, axis=a )
	p25 = np.percentile( data, 25, axis=a )
	p50 = np.percentile( data, 50, axis=a )
	p75 = np.percentile( data, 75, axis=a )
	p95 = np.percentile( data, 95, axis=a )

	# print( '{} {} {} {} {} {} {} {} {}'.format( mean.shape, var.shape, skewness.shape, kurt.shape, p5.shape, p25.shape, p50.shape, p75.shape, p95.shape ) )

	return np.hstack(( mean, var, skew, p5, p25, p50, p75, p95 ))	



def get_features( in_path ):

	X = np.array([])
	y = np.array([])

	num_features = 11

	print('Extracting features...')

	for file in sorted( os.listdir( in_path ) ):

		print( '\tProcessing {}'.format( file ) )

		rate, samples = wavfile.read( os.path.join( in_path, file ) )

		# Each ow corresponds to a sequence for same coefficient
		mfcc_mat = librosa.feature.mfcc( y=samples, sr=rate )
		fft = abs( scipy.fft( samples ) )

		curr_X = np.zeros([ 1, num_features ])
		curr_y = np.zeros([1])


		curr_X = calculate_stats( mfcc_mat, 1 )
		# curr_X.append()

		curr_y = label_mapping[file.split('_')[0]]

		if X.shape[0]:
			X = np.append( X, curr_X, axis=0 )
			y = np.append( y, curr_y )
			pass

		else:
			X = curr_X
			y = curr_y
			pass

		pass

	return X, y

def test_classifier( rf, X_test, y_test ):
	print('Testing the classifier...')
	y_hat = rf.predict( X_test )
	mat = confusion_matrix( y_test, y_hat )
	print( 'Confusion matrix:\n{}'.format( mat ) )

	print( 'Accuracy score: {}'.format( accuracy_score( y_test, y_hat ) ) )


	pass

def train_classifer( X_train, y_train ):

	print('Training a redom forest...')
	rf = RandomForestClassifier( n_estimators=50, n_jobs=-1 )
	rf.fit( X_train, y_train )

	return rf

def main():

	parser = OptionParser()
	parser.add_option( '-t', '--train', dest='train_mode', default=True, action='store_true', help='run in train mode' )
	parser.add_option( '-s', '--test', dest='train_mode', default=True, action='store_false', help='run in test mode' )
	parser.add_option( '-p', '--path', dest='in_path', default='', help='specify the input path' )

	( options, args ) = parser.parse_args()



	X, y = get_features( options.in_path )

	X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33 )

	rf = train_classifer( X_train, y_train )
	test_classifier( rf, X_test, y_test )




	

if __name__ == '__main__':
	main()

