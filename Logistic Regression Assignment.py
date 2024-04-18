# Lawrence Menegus 
# Machine Learning CPSC 429 
# Program Description: This program uses one input file to create a Logistic Regression Model
# First it will normalize the features, Then using the inital weights given to me in the file 
# it will calcualte the prediction, error, error squared, and the Delta Error W0 W1 W2, 
# Then the program will print out the New weights using the gradient decent algorithm alpha =0.02.
# lastly the program will update the weights each iteration up to 2000 and print out the weights.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as mlines

# Sigmoid Function
def sigmoid(z, derivative=False):
    s = 1 / (1 + np.exp(-z))

    # calculate the derivative
    if derivative:
        return s * (1 - s)
    else:
        return s

# Calculate results
def calculate_results(X, y, weights):
    print('\nTarget\t Pred\t\t Error\t\t errsqr\t\t w[0]\t\t w[1]\t\t w[2]')

    for i in range(min(5 , len(y))):
        target = y[i]
        pred = sigmoid(X[i].dot(weights))
        error = target - pred

        # Error Deltas
        Delta0 = error * sigmoid(X[i].dot(weights), derivative=True)
        Delta1 = error * sigmoid(X[i].dot(weights), derivative=True) * X[i,1]
        Delta2 = error * sigmoid(X[i].dot(weights), derivative=True) * X[i,2]

        # Printout for the output of the first 5
        print(f"{int(target)} \t {pred:.8f} \t {error:.8f} \t {error**2:.8f} \t {Delta0:.8f} \t {Delta1:.8f} \t {Delta2:.8f}")

# Normalize the values
def normalize_features(features):
    min_val = np.min(features, axis=0)
    max_val = np.max(features, axis=0)
    normalized_features = -1 + 2 * (features - min_val) / (max_val - min_val)
    return normalized_features

# Gradient Descent Function
def gradientDescent(X, y, weights, alpha, Iterations):
    m = y.size
    J_history = np.zeros(shape=(Iterations, 1))

    for i in range(Iterations):
        pred = sigmoid(X.dot(weights))
        error = y - pred

        # compute gradient
        gradient = X.T.dot(error * sigmoid(X.dot(weights), derivative=True))
        weights += alpha * gradient

        # compute and save cost
        J_history[i] = compute_cost(X, y, weights)


    return weights, J_history

# Define a function to compute the cost for the sum error squared for Logistic regression
def compute_cost(X, y, weights):
    m = y.size
    pred = sigmoid(X.dot(weights))
    error = y - pred
    errsqr = error**2

    # Cost function for Logistic regression 
    J =  np.sum(errsqr) / 2

    return J
    
# Plot Graph Function
def plot_graphs(X, y, weights, iteration):
    
    plt.figure()

    # Plot the positive examples as blue circles and red as X
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', s=50)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='x', s=50)

    # Get the minimum and maximum values of the features
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    # Create a grid of points with a 0.01 intervaland add bias to weights 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    weights_with_bias = np.array([weights[0], weights[1], weights[2]])

    # Compute the sigmoid function for each point in the grid
    Z = sigmoid(np.dot(np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()], weights_with_bias))

    # Reshape the result to match the grid shape
    Z = Z.reshape(xx.shape)

    # Draw the decision boundary as a straight line instead of a contour plot
    plt.contour(xx, yy, Z, levels=[0.5], colors='red', linewidths=2)

    # Create the legend label with the iteration number and the intercept and slope of the line
    legend_label = f'y ={weights[0]/weights[2]:.2f} + {weights[1]/weights[2]:.2f}r  (Iter={iteration})'
    plt.legend(handles=[mlines.Line2D([], [], color='red', label=legend_label)])
    plt.xlabel('RPM')
    plt.ylabel('VIBRATION')
    plt.show()


# Main Function
def main():
    # Input from Txt File
    data = np.genfromtxt('Table7_7.txt', delimiter=',')
    X = data[:, 1:3]
    y = data[:, 3]

    X = np.c_[np.ones(X.shape[0]), X]
    X_normalized = normalize_features(X[:, 1:])

    # Alpha and bias
    X_bias = np.c_[np.ones(X_normalized.shape[0]), X_normalized]
    alpha = 0.02

    # Print the first five rows of the Normalized Data
    print("\nFirst five rows of normalized data")
    print(X_normalized[:5, :])

    # Initial Weights
    initial_weights = [-2.9465, -1.0147, 2.161]
    print('\nInitially\nThe weights are:', ', '.join(map(str, initial_weights)))

    # Call the method to Print results
    calculate_results(X_bias, y, initial_weights)

    # Error Squared Calculations
    errsqr_sum = compute_cost(X_bias, y, initial_weights)
    print("\t\t\tThe Sum of squared errors sum/2 =", errsqr_sum)

    # Iteration 1
    iterations = 1
    new_weights, cost_history = gradientDescent(X_bias, y, initial_weights, alpha, iterations)
    print('After 1 iteration\nThe weights are:', ', '.join(map(lambda x: f"{x:.8f}", new_weights)))
    calculate_results(X_bias, y, new_weights)
    errsqr_sum = compute_cost(X_bias, y, new_weights)
    print("\t\t\tThe Sum of squared errors sum/2 =", errsqr_sum)
    plot_graphs(X_normalized, y, new_weights, 1)


    # Iteration 2
    iterations = 2
    new_weights, cost_history = gradientDescent(X_bias, y, initial_weights, alpha, iterations)
    print('After 2 iterations\nThe weights are:', ', '.join(map(lambda c: f"{c:.8f}", new_weights)))
     

    ### For Graphing ####
    # Iteration 10
    iterations = 10
    new_weights, cost_history = gradientDescent(X_bias, y, initial_weights, alpha, iterations)
    plot_graphs(X_normalized, y, new_weights, 10)

     # Iteration 200
    iterations = 200
    new_weights, cost_history = gradientDescent(X_bias, y, initial_weights, alpha, iterations)
    plot_graphs(X_normalized, y, new_weights, 200)

     # Iteration 500
    iterations = 500
    new_weights, cost_history = gradientDescent(X_bias, y, initial_weights, alpha, iterations)
    plot_graphs(X_normalized, y, new_weights, 500)


    # Iteration 2000
    iterations = 2000
    new_weights, cost_history = gradientDescent(X_bias, y, initial_weights, alpha, iterations)
    calculate_results(X_bias, y, new_weights)
    errsqr_sum = compute_cost(X_bias, y, new_weights)
    print("\t\t\tThe Sum of squared errors sum/2 =", errsqr_sum)
    print('After 2000 iterations\nThe weights are:', ', '.join(map(lambda b: f"{b:.8f}", new_weights)))
    plot_graphs(X_normalized, y, new_weights, 2000)

    # Cost function and iterations plot with limited y-axis range
    plt.figure()
    plt.plot( cost_history, color='red', linestyle='-')
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.show()

main()
