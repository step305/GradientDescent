import numpy as np
import inspect

gradient_step = 0.003
method_coefficient = 0.1


def goal_function(x, parameters):
    a = parameters[0]
    b = parameters[1]
    c = parameters[2]
    return a * x ** 2 + b * x + c


def gradient(x, p):
    delta_x = abs(gradient_step * x)
    f_left = goal_function(x - delta_x, p)
    f_right = goal_function(x + delta_x, p)
    return (f_right - f_left) / delta_x


if __name__ == '__main__':
    print('Gradient Descent method for optimization problems')

    print('Let we have goal function: f(x) = a * x^2 + b * x + c which value should be minimized')
    print()
    print('*' * 60)
    print(inspect.getsource(goal_function))
    print('*' * 60)

    print('Let define function for gradient calculation:')
    print()
    print('*' * 60)
    print(inspect.getsource(gradient))
    print('*' * 60)

    x0 = 5.0
    print('Now assume initial conditions:')
    print('Initial parameter of our model is: ', x0)

    model_p = [2, 3, -6]
    print('function parameters (polynomial coefficients): ', model_p)

    print("Let's make some steps:")

    x_prediction = x0
    for i in range(10):
        print('x = {:10.4f}, f(x) = {:10.4f}, gradient = {:10.4f}'.format(x_prediction,
                                                                          goal_function(x_prediction, model_p),
                                                                          gradient(x_prediction, model_p)))
        x_prediction = x_prediction - method_coefficient * gradient(x_prediction, model_p)

    print()
    print('*' * 60)
    print()
    print('Latest prediction:')
    print('x = {:10.4f}, f(x) = {:10.4f}, gradient = {:10.4f}'.format(x_prediction,
                                                                      goal_function(x_prediction, model_p),
                                                                      gradient(x_prediction, model_p)))

    print('True minimum for function is: {:10.4f}'.format(-model_p[1] / 2.0 / model_p[0]))
