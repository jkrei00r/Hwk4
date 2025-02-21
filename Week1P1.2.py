#Revised first assignment
# region imports
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
# endregion

# region function definitions
def Probability(PDF, args, c, GT=True):
    """
    This is the function to calculate the probability that x is >c or <c depending
    on the GT boolean.
    :param PDF: the probability density function to be integrated
    :param args: a tuple with (mean, standard deviation)
    :param c: value for which we ask the probability question
    :param GT: boolean deciding if we want probability x>c (True) or x<c (False)
    :return: probability value
    """
    mu, sig = args
    lhl = mu - 5 * sig
    rhl = c
    p, _ = quad(PDF, lhl, rhl, args=(mu, sig))
    return 1 - p if GT else p

def GPDF(x, mu, sig):
    """
    Gaussian probability density function.
    :param x: value at which to evaluate the PDF
    :param mu: mean of the distribution
    :param sig: standard deviation of the distribution
    :return: value of GPDF at the desired x
    """
    return (1 / (sig * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sig) ** 2)

def main():
    # Testing Probability with quad
    p1 = Probability(GPDF, (0, 1), 0, True)
    print("p1={:0.5f}".format(p1))  # Should be 0.5

    p2 = 1 - 2 * Probability(GPDF, (0, 1), 1)
    print("p2={:0.5f}".format(p2))  # Should be ~0.6826

    p3 = 1 - 2 * Probability(GPDF, (0, 1), 2)
    print("p3={:0.5f}".format(p3))  # Should be ~0.9544

    p4 = 1 - 2 * Probability(GPDF, (0, 1), 3)
    print("p4={:0.5f}".format(p4))  # Should be ~0.9974

# region function calls
if __name__ == '__main__':
    main()
# endregion