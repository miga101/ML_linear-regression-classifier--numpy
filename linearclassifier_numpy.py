# implementation from scratch of a linear classifier
import numpy as np


def compute_error4_line_given_points(b, m, points):
    total_error = 0
    # for all points: SUM UP
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (x*m + b)) **2
    #average
    return total_error/float(len(points))
    

def gradient_descent_runner(points, start_b, start_m, learning_rate, iterations):
    b = start_b
    m = start_m
    for i in range(iterations):
        # update m, b values with less error
        # performing one gradient step
        b, m = gradiant_step(b, m, np.array(points), learning_rate)
        #print('b= ', b)
    return [b, m]
        	
        	
def gradiant_step(current_b, current_m, points, learning_rate):     	
    gradient_b = 0
    gradient_m = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        gradient_b += -(2/N)*(y-((x*current_m) + current_b))
        gradient_m += -(2/N)*x*(y-((x*current_m) + current_b))
    
    new_b = current_b - (gradient_b*learning_rate)
    new_m = current_m - (gradient_m*learning_rate)
    
    return [new_b, new_m]
    

## 1. Get dataset
points = np.genfromtxt('storage/emulated/0/data_estudents.csv',
	    delimiter=',')
#print(points)

## 2. Define hyperparameters
# how fast should the model converge
learning_rate = 0.0001
# line equation: y = x*m + b
_b = 0
_m = 0
iterations = 1000

## 3. Train model
error = compute_error4_line_given_points(_b, _m, points)
#print('starting gradient descent at b={0}, m={1}, error={2}'.
#	format(_b, _m, compute_error4_line_given_points(_b, _m, points))
print('starting gradient descent at b={0}, m={1}, error={2}'.
	format(_b, _m, error))

print("Running...")
[b, m] = gradient_descent_runner(points, _b, _m, learning_rate,
	iterations)
	
print("After {0} iterations b = {1}, m = {2}, error = {3}".
	format(iterations, b, m, 
		compute_error4_line_given_points(b, m, points)))



