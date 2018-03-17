# author: Yerbol Aussat

import numpy as np
import matplotlib.pyplot as plt

# loading training data
def load_samples():
	global trn_data
 	trn_data = np.genfromtxt('spambase_X.csv', delimiter=',')
	
	
				
# loading labels
def load_labels():
	global trn_labels
 	trn_labels = np.genfromtxt('spambase_y.csv', delimiter=',')

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
	
	w = np.zeros((trn_data.shape[1], 1))
	b = 0
	max_pass = 500
	
	w, b, mistakes = perceptron_learning(trn_data, trn_labels, w, b, 500)
# 	print w
# 	print b
# 	print mistakes
	plt.plot(range(len(mistakes)), mistakes)
	plt.title('Perceptron Algorithm')
	plt.ylabel('Number of mistakes')
	plt.xlabel('Number of passes')
	plt.show()

if __name__ == "__main__":
    main()

