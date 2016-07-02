import numpy as np
import matplotlib.pyplot as plt


## proximal gradient optimization
def proximal_gradient(objective_func, smooth_func, smooth_gradient, smooth_linear, simple_func, x, step_size, beta, max_iter = 100):
	count = 0
	xs = [x]
	ys = [objective_func(x)]

	while count < max_iter:
		while True:
			z = simple_func(x - step_size*smooth_gradient(x), step_size)
			count += 1
			if smooth_func(z) <= smooth_linear(z, x, step_size) or count > max_iter:
				break
			step_size *= beta
		x = z
		xs.append(x)
		ys.append(objective_func(x))
	return np.squeeze(xs), ys

##	Objective function
def func_generator(A, b, c, l):
	def objective_func(x):
		return np.squeeze(smooth_func(x) + l*np.sum(np.abs(x)))

	def smooth_func(x):
		return np.squeeze(0.5*np.dot(np.dot(x.T, A), x) + np.dot(b.T, x) + c)

	def smooth_gradient(x):
		return 0.5*np.dot((A+A.T), x) + b

	def smooth_linear(x, y, lam):
		return np.squeeze(smooth_func(y) + np.dot(smooth_gradient(y).T, x-y) + 1/(2*lam)*np.dot((x-y).T, x-y))

	def prox_operator(x, step_size):
		return np.maximum(0, 1 - l*step_size/np.abs(x)) * x;

	return objective_func, smooth_func, smooth_gradient, smooth_linear, prox_operator


## Experiment

# Generate functions and run proximal gradient optimization
objective_func, smooth_func, smooth_gradient, smooth_linear, prox_operator = func_generator(np.array([[2, 0.25], [0.25, 0.2]]), np.array([[0.5, 0.5]]).T, -1.5, 0.2)
xs, ys = proximal_gradient(objective_func, smooth_func, smooth_gradient, smooth_linear, prox_operator, np.array([[0,4]]).T, 1, 0.9, 20)

# Draw contour map
delta = 0.05
x = np.arange(-4.0, 4.0, delta)
y = np.arange(-4.0, 4.0, delta)
X, Y = np.meshgrid(x, y)
Z = np.array([[objective_func(np.array([[x, y]]).T) for x,y in zip(row_x, row_y)] for row_x, row_y in zip(X,Y)])
plt.figure()
CS = plt.contour(X, Y, Z, 30)

# plot trajectory
plt.plot(xs[:,0], xs[:,1], 'r-')
plt.plot(xs[:,0], xs[:,1], 'gx')

# plot objective values w.r.t. iterations
plt.figure()
plt.plot(ys, 'r-')
plt.plot(ys, 'gx')

plt.show()