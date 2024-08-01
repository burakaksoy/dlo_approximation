import numpy as np
import matplotlib.pyplot as plt

def generate_weighting(n, alpha=10):
    """
    Generate a discrete weighting function where the last element has a very high weight
    and the first element has a very low weight. The sum of all weights is normalized to 1.
    
    Parameters:
    n (int): Number of discrete sections.
    alpha (float): Common ratio of the geometric series, must be greater than 1.
    
    Returns:
    weights (np.ndarray): Normalized weights.
    """
    # Generate the weights using a geometric series
    weights = np.array([alpha**i for i in range(n)], dtype=float)
    
    # Normalize the weights to sum up to 1
    weights /= np.sum(weights)
    
    return weights

# Example usage
n = 3
weights = generate_weighting(n)

print("weights = ", weights)

# Plotting the weights
plt.figure(figsize=(10, 6))
plt.stem(range(n), weights, basefmt=" ", use_line_collection=True)
plt.xlabel('Discrete Section')
plt.ylabel('Weight')
plt.title('Discrete Weighting Function')
plt.grid(True)
plt.show()

# ------------------------------------------------------------------------------

def generate_middle_peak_weighting_function(n, sigma=1.0):
    """
    Generate a discrete weighting function where the middle element has the highest weight
    and the first and last elements have the lowest weight. The sum of all weights is normalized to 1.
    
    Parameters:
    n (int): Number of discrete sections.
    sigma (float): Standard deviation of the Gaussian distribution.
    
    Returns:
    weights (np.ndarray): Normalized weights.
    """
    # Generate the x values centered around the middle of the range
    x = np.linspace(-1, 1, n)
    
    # Generate the weights using a Gaussian function
    weights = np.exp(-0.5 * (x / sigma)**2)
    
    # Normalize the weights to sum up to 1
    weights /= np.sum(weights)
    
    return weights

# Example usage
n = 3
weights = generate_middle_peak_weighting_function(n, sigma=0.4)

# Plotting the weights
plt.figure(figsize=(10, 6))
plt.stem(range(n), weights, basefmt=" ", use_line_collection=True)
plt.xlabel('Discrete Section')
plt.ylabel('Weight')
plt.title('Middle Peak Discrete Weighting Function')
plt.grid(True)
plt.show()
