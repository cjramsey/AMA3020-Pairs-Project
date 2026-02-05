import numpy as np

from time import perf_counter
from utils import (partial_derivative, second_partial_derivative, 
				   hessian, jacobian, gradient)


# Variations of Newton's method

def n_dim_newton_method(funcs, inital_guess, tol=1e-10, max_iter=100, history=False):
	'''
	Solve a system of n nonlinear equations using the Newton-Raphson method.

	Parameters:
		funcs (List[Callable]): List of functions representing the system of equations.
		initial_guess (List | np.array) Initial guess for the solution.
		tol (float): Tolerance for convergence.
		max_iter (int): Maximum number of iterations.
		history (bool): If true, return all steps in approximation.

	Returns:
		(np.ndarray): Array of solution approximations at each iteration if history=True,
				  otherwise returns the final solution approximation.
	'''
	x = np.array(inital_guess, dtype=np.float64)
	path = [x.copy()]
	k = 1

	while k <= max_iter:
		F = np.array([f(x) for f in funcs], dtype=np.float64)
		J = jacobian(funcs, x)

		try:
			y = np.linalg.solve(J, -F)
		except np.linalg.LinAlgError:
			raise ValueError("Jacobian is singular or not square.")

		x = x + y

		path.append(x.copy())

		# Stopping criterion: norm of difference in approximations < tol
		if np.linalg.norm(y, ord=float("inf")) < tol:
			if not history:
				return x
			return np.array(path)

		k += 1
	
	raise StopIteration("Maximum number of iterations reached without convergence.")


def newton_method_optimization(func, initial_guess, tol=1e-10, max_iter=100, history=False):
	'''
	Minimize a multivariable function using Newton's method.

	Parameters:
		func (Callable): The multivariable function to minimize.
		initial_guess (List | np.array): Initial guess for the minimum.
		tol (float): Tolerance for convergence.
		max_iter (int): Maximum number of iterations.
		history (bool): If true, return all steps in approximation.


	Returns:
		(np.ndarray): Array of solution approximations at each iteration if history=True,
				  	  otherwise returns the final solution approximation.
	'''
	x = np.array(initial_guess, dtype=np.float64)
	path = [x.copy()]
	k = 1

	while k <= max_iter:
		grad = gradient(func, x)
		H = hessian(func, x)

		try:
			y = np.linalg.solve(H, -grad)
		except np.linalg.LinAlgError:
			raise ValueError("Hessian is singular or not square.")
		
		x = x + y

		path.append(x.copy())

		if np.linalg.norm(grad, ord=float("inf")) < tol:
			if not history:
				return x
			return np.array(path)

		k += 1
	
	raise StopIteration("Maximum number of iterations reached without convergence.")



def newton_method_optimization_with_line_search(func, initial_guess, tol=1e-10, max_iter=100, history=False):
	'''
	Minimize a multivariable function using the Newton's method with backtracking line-search.

	Parameters:
		func (Callable): The multivariable function to minimize.
		initial_guess (List | np.array): Initial guess for the minimum.
		tol (float): Tolerance for convergence.
		max_iter (int): Maximum number of iterations.
		history (bool): If true, return all steps in approximation.


	Returns:
		(np.ndarray): Array of solution approximations at each iteration if history=True,
				  	  otherwise returns the final solution approximation.
	'''
	x = np.array(initial_guess, dtype=np.float64)
	path = [x.copy()]
	k = 1

	while k <= max_iter:
		grad = gradient(func, x)
		H = hessian(func, x)

		# Stopping criterion: norm of gradient < tol
		if np.linalg.norm(grad, ord=float("inf")) < tol:
			if not history:
				return x
			return np.array(path)

		try:
			p = np.linalg.solve(H, -grad)
		except np.linalg.LinAlgError:
			raise ValueError("Hessian is singular or not square.")
		
		# Backtracking line-search

		alpha = 1.0 	# Start with the full Newton step
		rho = 0.5     	# Shrink factor (common values: 0.1 to 0.5)
		c = 1e-4 		# Armijo condition constant (sufficient decrease)
		
		# While the new value doesn't meet the Armijo condition, shorten the step
		current_f = func(x)
		while func(x + alpha * p) > current_f + c * alpha * np.dot(grad, p):
			alpha *= rho
			if alpha < 1e-8: # Safety break to prevent infinite loop
				break
		
		y = alpha * p
		x += y

		path.append(x.copy())

		k += 1
	
	raise StopIteration("Maximum number of iterations reached without convergence.")


# Steepest Descent variations

def steepest_descent(func, initial_guess, max_iter=1000, tol=1e-6, ascent=False, alpha=0.1, 
					 require_conv=False, history=False):
	'''
	Minimize a multivariable function using the Steepest Descent/Ascent method with
	with constant alpha.

	Parameters:
		func (Callable): The multivariable function to minimize (or maximise).
		initial_guess (List | np.array): Initial guess for the minimum.
		max_iter (int): Maximum number of iterations.
		tol (float): Tolerance for convergence.
		ascent (bool): If True, performs steepest ascent instead of descent.
		alpha (float): Step size for each iteration.
		require_conv (bool): Whether exceeding max_iter iterations raises exception or returns approximation.
		history (bool): If true, return all steps in approximation.

	Returns:
		(np.ndarray): Array of solution approximations at each iteration if history=True,
					  otherwise returns the final solution approximation.
	'''
	x = np.array(initial_guess, dtype=np.float64)
	path = [x.copy()]
	k = 1

	while k <= max_iter:
		grad = gradient(func, x)

		if np.linalg.norm(grad, ord=float("inf")) < tol:
			if not history:
				return x
			return np.array(path)

		if ascent:
			x = x + alpha * grad
		else:
			x = x - alpha * grad
			
		path.append(x.copy())

		k += 1
	
	if not require_conv:
		if not history:
			return x
		return np.array(path)
	
	raise StopIteration("Maximum number of iterations reached without convergence.")


def steepest_descent_with_line_search(func, initial_guess, max_iter=1000, tol=1e-6, 
					 require_conv=False, history=False):
	'''
	Minimize a multivariable function using the Steepest Descent/Ascent method using
	backtracking line-search.

	Parameters:
		func (Callable): The multivariable function to minimize (or maximise).
		initial_guess (List | np.array): Initial guess for the minimum.
		max_iter (int): Maximum number of iterations.
		tol (float): Tolerance for convergence.
		alpha (float): Step size for each iteration.
		require_conv (bool): Whether exceeding max_iter iterations raises exception or returns approximation.
		history (bool): If true, return all steps in approximation.

	Returns:
		(np.ndarray): Array of solution approximations at each iteration if history=True,
				  	  otherwise returns the final solution approximation.
	'''
	x = np.array(initial_guess, dtype=np.float64)
	path = [x.copy()]
	k = 1

	while k <= max_iter:
		grad = gradient(func, x)
		p = -grad

		if np.linalg.norm(grad, ord=float("inf")) < tol:
			return np.array(path) if history else x

		alpha = 1.0   
		rho = 0.5          # Shrink factor (common values: 0.1 to 0.5)
		c = 1e-4           # Armijo condition constant (sufficient decrease)
		
		# While the new value doesn't meet the Armijo condition, shorten the step
		# Condition: f(x + alpha*p) <= f(x) + c * alpha * grad.dot(p)
		current_f = func(x)
		while func(x + alpha * p) > current_f + c * alpha * np.dot(grad, p):
			alpha *= rho
			if alpha < 1e-12: # Safety break to prevent infinite loop
				break
		
		x += alpha * p
		path.append(x.copy())

		k += 1
	
	if not require_conv:
		return np.array(path) if history else x
	
	raise StopIteration("Maximum number of iterations reached without convergence.")


# Hybrid methods

def newton_optimization_with_steepest_descent(func, initial_guess, tol=1e-8, 
											  max_iter=1000, descent_iter=10, 
											  ascent=False, alpha=0.02, history=False):
	'''
	Minimize a multivariable function using the Newton-Raphson method, using the steepest
	descent/ascent algorithm to refine the starting approximation.

	Parameters:
		func (Callable): The multivariable function to minimize (or maximise).
		initial_guess (List | np.array): Initial guess for the minimum.
		tol (float): Tolerance for convergence.
		max_iter (int): Maximum number of iterations for Newton's method.
		descent_iter (int): Maximum number of iterations for steepest descent algorithm.
		ascent (bool): If True, performs steepest ascent instead of descent.
		alpha (float): Step size for each iteration.
		history (bool): If true, return all steps in approximation.

	Returns:
		(np.ndarray): Array of solution approximations at each iteration if history=True,
					  otherwise returns the final solution approximation.
	'''
	refined_guess = steepest_descent(func, initial_guess, max_iter=descent_iter, alpha=alpha, ascent=ascent)
	return newton_method_optimization(func, refined_guess, tol=tol, max_iter=max_iter, history=history)


# Basins of attraction with various methods

def find_basins_of_attraction(func, roots, res=300, xlim=(-2, 2), ylim=(-2, 2)):
	'''
	Find basins of attraction for a given function on a specified grid using
	Newton's method.

	Parameters:
		func (Callable): The multivarible function.
		roots (List | np.array): The critical points (i.e the roots of the gradient).
		res (int): The resolution of the grid.
		xlim (Tuple[int, int]): Limits of the x-axis.
		ylim (Tuple[int, int]): Limits of the y-axis.

	Returns:
		(Tuple[np.ndarray, np.array, np.array]): Grid of integers corresponding to critical points.
	'''
	x = np.linspace(xlim[0], xlim[1], res)
	y = np.linspace(ylim[0], ylim[1], res)
	X, Y = np.meshgrid(x, y)
	basins = np.zeros(X.shape)

	for i in range(res):
		for j in range(res):
			point = np.array([X[i, j], Y[i, j]])
			try:
				solution = newton_method_optimization(func, point, history=False, tol=1e-6, max_iter=100)
			except Exception:
				pass
			else:
				final_point = np.array([solution[0], solution[1]])
				distances = [np.linalg.norm(final_point - root) for root in roots]
				basins[i, j] = np.argmin(distances) + 1  # +1 to avoid zero index
	
	return basins, X, Y


def find_basins_of_attraction_with_descent(f, roots, res=300, xlim=(-2, 2), ylim=(-2, 2), ascent=False):
	'''
	Find basins of attraction for a given function on a specified grid using
	Newton's method starting with Steepest Descent.

	Parameters:
		func (Callable): The multivarible function.
		roots (List | np.array): The critical points (i.e the roots of the gradient).
		res (int): The resolution of the grid.
		xlim (Tuple[int, int]): Limits of the x-axis.
		ylim (Tuple[int, int]): Limits of the y-axis.
		ascent (bool): If True, performs steepest ascent instead of descent.

	Returns:
		(Tuple[np.ndarray, np.array, np.array]): Grid of integers corresponding to critical points.
	'''
	x = np.linspace(xlim[0], xlim[1], res)
	y = np.linspace(ylim[0], ylim[1], res)
	X, Y = np.meshgrid(x, y)
	basins = np.zeros(X.shape)

	for i in range(res):
		for j in range(res):
			point = np.array([X[i, j], Y[i, j]])
			try:
				solution = newton_optimization_with_steepest_descent(f, point, tol=1e-6, max_iter=100, 
														 descent_iter=35, alpha=0.02, ascent=ascent)
			except Exception:
				pass
			else:
				final_point = np.array([solution[0], solution[1]])
				distances = [np.linalg.norm(final_point - root) for root in roots]
				basins[i, j] = np.argmin(distances) + 1  # +1 to avoid zero index
	
	return basins, X, Y


def find_basins_of_attraction_with_line_search(f, roots, res=300, xlim=(-2, 2), ylim=(-2, 2)):
	'''
	Find basins of attraction for a given function on a specified grid using
	Newton's method with line-search.
	
	Parameters:
		func (Callable): The multivarible function.
		roots (List | np.array): The critical points (i.e the roots of the gradient).
		res (int): The resolution of the grid.
		xlim (Tuple[int, int]): Limits of the x-axis.
		ylim (Tuple[int, int]): Limits of the y-axis.
		ascent (bool): If True, performs steepest ascent instead of descent.

	Returns:
		(Tuple[np.ndarray, np.array, np.array]): Grid of integers corresponding to critical points.
	'''
	x = np.linspace(xlim[0], xlim[1], res)
	y = np.linspace(ylim[0], ylim[1], res)
	X, Y = np.meshgrid(x, y)
	basins = np.zeros(X.shape)

	for i in range(res):
		for j in range(res):
			point = np.array([X[i, j], Y[i, j]])
			try:
				solution = newton_method_optimization_with_line_search(f, point, history=False, tol=1e-6, max_iter=100)
			except Exception:
				pass
			else:
				final_point = np.array([solution[0], solution[1]])
				distances = [np.linalg.norm(final_point - root) for root in roots]
				basins[i, j] = np.argmin(distances) + 1  # +1 to avoid zero index
	
	return basins, X, Y


if __name__ == "__main__":
	pass