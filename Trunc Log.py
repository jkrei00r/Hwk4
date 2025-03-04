# Truncated log-normal distribution
# region imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
# endregion

# region function definitions
def truncated_lognormal_pdf(x, mu, sigma, lower, upper):
    """
    Compute the PDF of a truncated log-normal distribution.

    Args:
        x (float or np.ndarray): Value(s) at which to evaluate the PDF.
        mu (float): Mean of the underlying normal distribution.
        sigma (float): Standard deviation of the underlying normal distribution.
        lower (float): Lower truncation limit.
        upper (float): Upper truncation limit.

    Returns:
        float or np.ndarray: Value(s) of the truncated log-normal PDF at x.
    """
    if x < lower or x > upper:
        return 0
    else:
        # Normalize the log-normal PDF by the truncation limits
        normalization = lognorm.cdf(upper, sigma, scale=np.exp(mu)) - lognorm.cdf(lower, sigma, scale=np.exp(mu))
        return lognorm.pdf(x, sigma, scale=np.exp(mu)) / normalization

def main():
    """
    Main function to generate and plot the truncated log-normal PDF and CDF.

    This function:
    1. Prompts the user for input parameters (mu, sigma, lower, and upper limits).
    2. Validates the input to ensure the lower limit is less than the upper limit.
    3. Computes the PDF and CDF of the truncated log-normal distribution.
    4. Plots the PDF with a shaded area representing the probability up to a specific upper limit.
    5. Plots the CDF.
    6. Displays the plots with proper labels and annotations.
    """
    # User input for log-normal distribution parameters
    mu = float(input("Enter the mean (mu) of the log-normal distribution: "))
    sigma = float(input("Enter the standard deviation (sigma) of the log-normal distribution: "))

    # Validate lower and upper truncation limits
    while True:
        lower = float(input("Enter the lower truncation limit: "))
        upper = float(input("Enter the upper truncation limit: "))
        if lower < upper:
            break
        else:
            print("Error: The lower limit must be less than the upper limit. Please try again.")

    # Generate x values for the plot
    x = np.linspace(lower, upper, 1000)

    # Calculate the PDF values for the truncated log-normal distribution
    pdf_values = np.array([truncated_lognormal_pdf(xi, mu, sigma, lower, upper) for xi in x])

    # Calculate the CDF values for the truncated log-normal distribution
    normalization = lognorm.cdf(upper, sigma, scale=np.exp(mu)) - lognorm.cdf(lower, sigma, scale=np.exp(mu))
    cdf_values = lognorm.cdf(x, sigma, scale=np.exp(mu)) / normalization

    # Determine the upper limit for the shaded area (75% of the range)
    D_upper = lower + (upper - lower) * 0.75

    # Calculate the shaded area (probability P(D ≤ D_upper))
    shaded_area = lognorm.cdf(D_upper, sigma, scale=np.exp(mu)) / normalization

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
    ax1.text(D_upper, max(pdf_values) * 0.5, f'P(D ≤ {D_upper:.2f}) = {shaded_area:.3f}', fontsize=12, color='black', ha='right')

    # Add the equation above the shaded area
    equation = r'$f(D)=\frac{1}{D\cdot \sigma \sqrt{2\pi }}\cdot e^{-\frac{1}{2}\left(\frac{\ln (D)-\mu }{\sigma }\right)^{2}}$'
    ax1.text((lower + upper) / 2, max(pdf_values) * 0.9, equation, fontsize=12, color='black', ha='center')

    # CDF plot
    ax2.plot(x, cdf_values, label='CDF: $F(D) = P(D \\leq d)$', color='orange')
    ax2.set_xlabel('D')
    ax2.set_ylabel('F(D)')
    ax2.set_title('Truncated Log-Normal CDF')
    ax2.legend()

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

# region function calls
if __name__ == '__main__':
    main()
# endregion
if __name__ == '__main__':
    main()
# endregion
