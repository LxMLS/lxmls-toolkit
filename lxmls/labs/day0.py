import numpy as np
import math
import matplotlib.pyplot as plt


# Exercise about gradient descent

def get_y(x):
    value = pow((x+2), 2) - 16*math.exp(-pow((x-2), 2))
    return value


def get_grad(x):
    return (2*x + 4) - 16*(-2*x + 4)*np.exp(-((x-2)**2))


def gradient_descent(start_x, func, grad):
    # Precision of the solution
    prec = 0.0001
    # Use a fixed small step size
    step_size = 0.1
    # max iterations
    max_iter = 100
    x_new = start_x
    res = []
    for i in xrange(max_iter):
        x_old = x_new
        # Use beta iguals to -1 for gradient descent
        x_new = x_old - step_size*grad(x_new)
        f_x_new = func(x_new)
        f_x_old = func(x_old)
        res.append([x_new, f_x_new])
        if abs(f_x_new - f_x_old) < prec:
            print "change in function values to small, leaving"
            return np.array(res)
    print "exceeded maximum number of iterations, leaving"
    return np.array(res)


def show_optimization_exercise():
    x = np.arange(-8, 8, 0.001)
    y = map(lambda u: get_y(u), x)
    plt.plot(x, y)
    x_0 = -8
    res = gradient_descent(x_0, get_y, get_grad)
    plt.plot(res[:, 0], res[:, 1], '+')
    x_0 = 8
    res = gradient_descent(x_0, get_y, get_grad)
    plt.plot(res[:, 0], res[:, 1], '*')
    plt.show()
