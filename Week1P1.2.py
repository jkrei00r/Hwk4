#Revised program from Week 1
# region imports
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
# endregion

# region function definitions
def Probability(PDF, args, c, GT=True):
    """
    Calculate the probability that x is greater than or less than a given value c.

    Args:
        PDF (callable): The probability density function to be integrated.
        args (tuple): A tuple containing the mean (mu) and standard deviation (sigma) of the distribution.
        c (float): The value for which the probability is calculated.
        GT (bool, optional): If True, calculates P(x > c). If False, calculates P(x < c). Defaults to True.

    Returns:
        float: The calculated probability.
    """
    mu, sig = args  # Unpack mean and standard deviation
    lhl = mu - 5 * sig  # Lower limit of integration (5 standard deviations below the mean)
    rhl = c  # Upper limit of integration

    # Integrate the PDF from lhl to rhl
    p, _ = quad(PDF, lhl, rhl, args=(mu, sig))

    # Return P(x > c) if GT is True, otherwise P(x < c)
    return 1 - p if GT else p

def GPDF(x, mu, sig):
    """
    Gaussian (Normal) Probability Density Function.

    Args:
        x (float): The value at which to evaluate the PDF.
        mu (float): The mean of the distribution.
        sig (float): The standard deviation of the distribution.

    Returns:
        float: The value of the Gaussian PDF at x.
    """
    return (1 / (sig * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sig) ** 2)

def find_root(PDF, args, c, target_probability, GT=True):
    """
    Find the root of the equation P(x > c) = target_probability using fsolve.

    Args:
        PDF (callable): The probability density function to be integrated.
        args (tuple): A tuple containing the mean (mu) and standard deviation (sigma) of the distribution.
        c (float): The initial guess for the root.
        target_probability (float): The target probability for which to solve.
        GT (bool, optional): If True, calculates P(x > c). If False, calculates P(x < c). Defaults to True.

    Returns:
        float: The value of c that satisfies P(x > c) = target_probability.
    """
    # Define the equation to solve: P(x > c) - target_probability = 0
    def equation(c_val):
        return Probability(PDF, args, c_val, GT) - target_probability

    # Use fsolve to find the root
    root = fsolve(equation, c)[0]
    return root

def main():
    """
    Main function to test the Probability function and find roots using fsolve.

    This function:
    1. Tests the Probability function with the Gaussian PDF for specific values of c.
    2. Uses fsolve to find the value of c that satisfies P(x > c) = target_probability.
    """
    # Test the Probability function with the Gaussian PDF
    p1 = Probability(GPDF, (0, 1), 0, True)
    print("p1 = P(x > 0) = {:0.5f}".format(p1))  # Should be 0.5

    p2 = 1 - 2 * Probability(GPDF, (0, 1), 1)
    print("p2 = P(-1 < x < 1) = {:0.5f}".format(p2))  # Should be ~0.6826

    p3 = 1 - 2 * Probability(GPDF, (0, 1), 2)
    print("p3 = P(-2 < x < 2) = {:0.5f}".format(p3))  # Should be ~0.9544

    p4 = 1 - 2 * Probability(GPDF, (0, 1), 3)
    print("p4 = P(-3 < x < 3) = {:0.5f}".format(p4))  # Should be ~0.9974

    # Use fsolve to find the value of c that satisfies P(x > c) = target_probability
    target_probability = 0.1  # Example target probability
    initial_guess = 1.0  # Initial guess for fsolve
    root = find_root(GPDF, (0, 1), initial_guess, target_probability, GT=True)
    print(f"Value of c for P(x > c) = {target_probability}: {root:.5f}")

# region function calls
if __name__ == '__main__':
    main()
# endregion
