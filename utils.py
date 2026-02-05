from datetime import datetime
import os
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np


def partial_derivative(func, var_index, point, h=1e-5):
	'''
	Compute the partial derivative of a multivariable function at a given point using
	2-point central difference.

	Parameters:
		func (Callable): The multivariable function.
		var_index (int): The index of the variable to differentiate with respect to.
		point (List | np.array): The point at which to compute the derivative.
		h (float): A small step size for finite difference approximation.

	Returns:
		(float): The partial derivative of the function at the given point.
	'''
	point_forward = np.array(point, dtype=np.float64)
	point_backward = np.array(point, dtype=np.float64)

	point_forward[var_index] += h
	point_backward[var_index] -= h

	return (func(point_forward) - func(point_backward)) / (2 * h)


def gradient(func, x):
	'''
	Compute the gradient of a multivariable function at a specified point.
	
	Parameters:
		func (Callable): The multivariable function.
		point (List | np.array): The point at which to compute the derivative.

	Returns:
		(np.ndarray): The gradient vector evaluated at the given point.
	'''
	return np.array([partial_derivative(func, i, x) for i in range(len(x))], dtype=np.float64)


def jacobian(funcs, point):
	'''
	Compute the Jacobian matrix of a system of functions at a given point.

	Parameters:
		funcs (List[Callable]): List of functions representing the system of equations.
		point (List | np.array): The point at which to compute the Jacobian.

	Returns:
		(np.ndarray): The Jacobian matrix evaluated at the given point.
	'''
	n = len(funcs)
	J = np.zeros((n, n), dtype=np.float64)

	for i, func in enumerate(funcs):
		for j in range(n):
			J[i, j] = partial_derivative(func, j, point)

	return J


def second_partial_derivative(func, var_index1, var_index2, point, h=1e-5):
	'''
	Compute the second partial derivative of a multivariable function at a given point
	using 4-point central difference.

	Parameters:
		func (Callable): The multivariable function.
		var_index1 (int): The index of the first variable to differentiate with respect to.
		var_index2 (int): The index of the second variable to differentiate with respect to.
		point (List | np.array): The point at which to compute the derivative.
		h (float): A small step size for finite difference approximation.

	Returns:
		(float): The second partial derivative of the function at the given point.
	'''
	if var_index1 != var_index2:
		point_pp = np.array(point, dtype=np.float64)
		point_pm = np.array(point, dtype=np.float64)
		point_mp = np.array(point, dtype=np.float64)
		point_mm = np.array(point, dtype=np.float64)

		point_pp[var_index1] += h
		point_pp[var_index2] += h

		point_pm[var_index1] += h
		point_pm[var_index2] -= h

		point_mp[var_index1] -= h
		point_mp[var_index2] += h

		point_mm[var_index1] -= h
		point_mm[var_index2] -= h

		return (func(point_pp) - func(point_pm) - func(point_mp) + func(point_mm)) / (4 * h**2)

	# Use different formula if variable indexes are the same
	else:
		pointp = np.array(point, dtype=np.float64)
		point = np.array(point, dtype=np.float64)
		pointm = np.array(point, dtype=np.float64)

		pointp[var_index1] += h
		pointm[var_index1] -= h

		return (func(pointp) - 2 * func(point) + func(pointm)) / (h**2)


def hessian(func, point):
	'''
	Compute the Hessian matrix of a multivariable function at a given point.

	Parameters:
		func (Callable): The multivariable function.
		point (List | np.array): The point at which to compute the Hessian.

	Returns:
		(np.ndarray): The Hessian matrix evaluated at the given point.
	'''
	n = len(point)
	H = np.zeros((n, n), dtype=np.float64)

	for i in range(n):
		for j in range(n):
			H[i, j] = second_partial_derivative(func, i, j, point)

	return H


def save_figure(fig, filename='', directory='plots'):
	'''
	Save a matplotlib figure with a timestamped filename.

	Parameters:
		fig (Figure): The figure to save.
		directory (str): The directory to save the figure in.

	Returns:
		(str): The path to the saved figure.
	'''
	# Check directory exists
	cur = os.getcwd()
	dir = os.path.join(cur, directory)
	if not os.path.exists(dir):
		os.makedirs(dir)

	# Create filepath using timestamp as file name
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	if not filename:
		filename = f"{timestamp}.pdf"
	filepath = os.path.join(dir, filename)
	fig.savefig(filepath)

	return filepath


def plot3d(func, x_lim, y_lim, num_points=100, alpha=0.8, cmap="viridis"):
	'''
	Generate a 3D surface plot of a function taking two variables.

	Parameters:
		func (Callable): The multivariable function to plot.
		x_lim (Tuple[float, float]): The limits for the x-axis.
		y_lim (Tuple[float, float]): The limits for the y-axis.
		num_points (int): Number of points along each axis.

	Returns:
		(Tuple[plt.figure, plt.axes]): The figure and axes.
	'''
	x = np.linspace(x_lim[0], x_lim[1], num_points)
	y = np.linspace(y_lim[0], y_lim[1], num_points)
	X, Y = np.meshgrid(x, y)
	Z = np.array([[func([xi, yi]) for xi in x] for yi in y])

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.plot_surface(X, Y, Z, cmap=cmap, alpha=alpha)

	# Default axes labels, can change after returning
	ax.set_xlabel("$X$")
	ax.set_ylabel("$Y$")

	return fig, ax


def test_method(method, func, points, root=None):
	'''
	Test performance of a given optimization algorithm on a given function and specified set of points.
	Prints the convergence rate, median iterations upon success, mean runtime upon success and 
	percentage of failures which occurred due to singular Hessian (only applicable to variations of
	Newton's method).

	Parameters:
		method (Callable): The algorithm being tested.
		func (Callable): The function being tested upon.
		points (List | np.ndarray): Set of points used to start iterations.
		roots (List | np.ndarray): Optionally specify root to prevent silent errors.

	Returns:
		None
	'''
	reached_max_iter = 0
	singular_hessian = 0
	diverged = 0
	success = 0
	iterations = []
	times = []

	for point in points:
		try:
			start = perf_counter()
			solution = method(func, point, history=True)
			if root and np.linalg.norm(root - solution[-1], ord=2) > 1:
				raise Exception
		except StopIteration:
			reached_max_iter += 1
		except ValueError:
			singular_hessian += 1
		except Exception:
			diverged += 1
		else:
			end = perf_counter()
			times.append(end - start)
			success += 1
			iterations.append(len(solution) - 1)

	total_failures = reached_max_iter + singular_hessian + diverged

	print(method.__name__)
	print(f"Conv. Rate: {success/len(points) * 100 if success else 0:.2f}%")
	print(f"% failures due to singular Hessian: {singular_hessian/total_failures * 100 if total_failures > 0 else 0:.2f}")
	print(f"Median Iterations: {np.median(iterations) if iterations else np.nan}")
	print(f"Avg Runtime (s): {np.mean(times) if times else np.nan:.6f}") 


if __name__ == "__main__":
	pass