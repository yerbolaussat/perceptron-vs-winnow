# author: Yerbol Aussat

import numpy as np
import matplotlib.pyplot as plt
import math
import collections

# loading training data
def load_samples():
	global trn_data
 	trn_data = np.genfromtxt('spambase_X.csv', delimiter=',')
	
# loading labels
def load_labels():
	global trn_labels
 	trn_labels = np.genfromtxt('spambase_y.csv', delimiter=',')

# Winnow
def balanced_winow(X, y, w_plus, w_minus, b_plus, b_minus, max_pass, nu):
	mistakes = []
	for t in range(max_pass):
		mistake = 0
		
		for i in range(X.shape[0]):	
	
			if ( np.dot(X[i], w_plus) + b_plus - np.dot(X[i], w_minus) - b_minus) * y[i] <= 0:				
				
				w_plus = np.multiply(w_plus, np.exp(nu * y[i] * X[i].reshape(X.shape[1], 1)))
				w_minus = np.multiply(w_minus, np.exp(- nu * y[i] * X[i].reshape(X.shape[1], 1)))
				
				b_plus = b_plus * math.exp(nu * y[i])
				b_minus = b_minus * math.exp(-nu * y[i])
				
				s_plus = b_plus + np.sum(w_plus)
				s_minus = b_minus + np.sum(w_minus)
								
				w_plus = w_plus/s_plus
				w_minus = w_minus/s_minus

				b_plus = b_plus/s_plus
				b_minus = b_minus/s_minus
				
				mistake += 1
# 		print mistake
		mistakes.append(mistake)
		if mistake == 0:
			break	
	return w_plus, b_plus, mistakes
	
# Perceptron
def perceptron_learning(X, y, w, b, max_pass):
	mistakes = []
	
	for t in range(max_pass):

		mistake = 0
		for i in range(X.shape[0]):	
			if (np.dot(X[i], w) + b) * y[i] <= 0:
				w = w + y[i] * X[i].reshape(X.shape[1], 1)
				b = b + y[i]
				mistake += 1
#  		print '\n iteration #', t
# 		print 'w', w
# 		print 'b', b	
#		print 'mistake', mistake	
		mistakes.append(mistake)
	return w, b, mistakes
	
	
def main():
	load_samples()
	load_labels()

 	trn_data_altered = np.concatenate((trn_data, np.random.standard_normal([trn_data.shape[0],100])), axis = 1)
	
	max_pass = 500
	nu = 1 / np.amax(trn_data_altered) # step size
	print nu
	
	w_plus = np.ones((trn_data_altered.shape[1], 1))  / (trn_data.shape[1] + 1)
	w_minus = np.ones((trn_data_altered.shape[1], 1)) / (trn_data.shape[1] + 1)
	b_plus = 1.0 / (trn_data_altered.shape[1] + 1)
	b_minus = 1.0 / (trn_data_altered.shape[1] + 1)
	w, b, mistakes = balanced_winow(trn_data_altered, trn_labels, w_plus, w_minus, b_plus, b_minus, max_pass, nu)
	
	w_init_p= np.zeros((trn_data_altered.shape[1], 1))
	b_init_p = 0
	w_p, b_p, mistakes_p = perceptron_learning(trn_data_altered, trn_labels, w_init_p, b_init_p, max_pass)
	
	
	plt.plot(range(len(mistakes)), mistakes, 'r-', label = "Winnow(nu = 1 / max element of A) = 6.3E-5")
	plt.plot(range(len(mistakes_p)), mistakes_p, 'b--', label = "Perceptron")
	
	plt.title('Winnow and Perceptron')
	plt.ylabel('Number of mistakes')
	plt.xlabel('Number of passes')
	plt.legend()
	plt.show()

if __name__ == "__main__":
    main()