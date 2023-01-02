import numpy as np
import inspect

gradient_step = 0.001
method_coefficient = 0.01


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

    model_p = [23, 12, -6]
    print('function parameters (polynomial coefficients): ', model_p)

    print("Let's make some steps:")

    x_prediction = x0
    f_previous = goal_function(x0, model_p)
    i = 0
    error = 10
    while error > 1e-8:
        print('step {:2d}: x = {:10.6f}, f(x) = {:10.6f}, gradient = {:10.6f}'.format(i,
                                                                                      x_prediction,
                                                                                      goal_function(x_prediction,
                                                                                                    model_p),
                                                                                      gradient(x_prediction, model_p)))
        x_prediction = x_prediction - method_coefficient * gradient(x_prediction, model_p)
        if (goal_function(x_prediction, model_p) - f_previous) < 1e-2 * f_previous:
            method_coefficient = method_coefficient / 2.0
        error = abs((f_previous - goal_function(x_prediction, model_p)) / f_previous)
        f_previous = goal_function(x_prediction, model_p)
        print('error: {:0.9f}'.format(error))
        i += 1

    print()
    print('*' * 60)
    print()
    print('Latest prediction:')
    print('x = {:10.6f}, f(x) = {:10.6f}, gradient = {:10.6f}'.format(x_prediction,
                                                                      goal_function(x_prediction, model_p),
                                                                      gradient(x_prediction, model_p)))

    print('True minimum for function is: {:10.6f}'.format(-model_p[1] / 2.0 / model_p[0]))
