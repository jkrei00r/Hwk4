# region imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.integrate import quad
# endregion

# region function definitions
def truncated_lognormal_pdf(x, mu, sigma, lower, upper):
    """
    Truncated log-normal PDF.
    :param x: value at which to evaluate the PDF
    :param mu: mean of the log-normal distribution
    :param sigma: standard deviation of the log-normal distribution
    :param lower: lower truncation limit
    :param upper: upper truncation limit
    :return: value of the truncated log-normal PDF at x
    """
    if x < lower or x > upper:
        return 0
    else:
        return lognorm.pdf(x, sigma, scale=np.exp(mu)) / (lognorm.cdf(upper, sigma, scale=np.exp(mu)) - lognorm.cdf(lower, sigma, scale=np.exp(mu)))

def main():
    # User input for log-normal distribution parameters
    mu = float(input("Enter the mean (mu) of the log-normal distribution: "))
    sigma = float(input("Enter the standard deviation (sigma) of the log-normal distribution: "))
    lower = float(input("Enter the lower truncation limit: "))
    upper = float(input("Enter the upper truncation limit: "))

    # Generate x values
    x = np.linspace(lower, upper, 1000)

    # Calculate PDF and CDF
    pdf_values = [truncated_lognormal_pdf(xi, mu, sigma, lower, upper) for xi in x]
    cdf_values = [quad(lambda xi: truncated_lognormal_pdf(xi, mu, sigma, lower, upper), lower, xi)[0] for xi in x]

    # Determine the upper limit for the shaded area
    D_upper = lower + (upper - lower) * 0.75
    shaded_area = quad(lambda xi: truncated_lognormal_pdf(xi, mu, sigma, lower, upper), lower, D_upper)[0]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # PDF plot
    ax1.plot(x, pdf_values, label='Truncated Log-Normal PDF')
    ax1.fill_between(x, pdf_values, where=(x <= D_upper), color='grey', alpha=0.5, label=f'P(D ≤ {D_upper:.2f}) = {shaded_area:.3f}')
    ax1.set_xlabel('D')
    ax1.set_ylabel('f(D)')
    ax1.set_title('Truncated Log-Normal PDF')
    ax1.legend()

    # Annotate the shaded area equation
    ax1.text(D_upper, max(pdf_values) * 0.5, f'P(D ≤ {D_upper:.2f}) = {shaded_area:.3f}', fontsize=12, color='black')

    # CDF plot
    ax2.plot(x, cdf_values, label='CDF: $F(D) = P(D \\leq d)$', color='orange')
    ax2.set_xlabel('D')
    ax2.set_ylabel('F(D)')
    ax2.set_title('Truncated Log-Normal CDF')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# region function calls
if __name__ == '__main__':
    main()
# endregion