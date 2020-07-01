import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
# Main Driver Function
if __name__ == '__main__':

	# Part c: Plot data and the optimal linear fit
	X = np.array([0, 2, 3, 4])
	y = np.array([1, 3, 6, 8])
	plt.style.use('ggplot')
	plt.plot(X, y, 'ro')
	# generate space for optimal linear fit
	m_opt = 62. / 35 # solution from part a
	b_opt = 18. / 35 # solution from part a
	X_space = np.linspace(-1, 5, num=100).reshape(-1, 1)
	y_space = (m_opt * X_space + b_opt).reshape(-1, 1)
	plt.plot(X_space, y_space)
	plt.savefig('hw1_pr2_c.png', format='png')
	plt.close()

    # Part d: Optimal linear fit with random data points
	# generate random data points
	mu, sigma, sampleSize = 0, 1, 100
	noise = np.random.normal(mu, sigma, sampleSize).reshape(-1, 1)
	# generate y-coordinate of the 100 points with noise
	y_space_rand = m_opt * X_space + b_opt + noise
	# calculate new weights
	X_space_stacked = np.hstack((np.ones_like(y_space), X_space))
	W_opt = np.linalg.solve(X_space_stacked.T @ X_space_stacked,
		X_space_stacked.T @ y_space_rand)
    # get the new m, and new b from W_opt obtained above
	b_rand_opt, m_rand_opt = W_opt.item(0), W_opt.item(1)
	# generate the y-coordinate of 100 points with the new parameters obtained
	y_pred_rand = np.array([m_rand_opt * x + b_rand_opt for x in X_space]).reshape(-1, 1)
	# generate plots with legend
	# plot original data points and line
	plt.plot(X, y, 'ro')
	orig_plot, = plt.plot(X_space, y_space, 'r')
	# plot the generated 100 points with white gaussian noise and the new line
	plt.plot(X_space, y_space_rand, 'bo')
	rand_plot, = plt.plot(X_space, y_pred_rand, 'b')
	# set up legend and save the plot to the current folder
	plt.legend((orig_plot, rand_plot), \
		('original fit', 'fit with noise'), loc = 'best')
	plt.savefig('hw1_pr2_d.png', format='png')
	plt.close()
