
import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 0.5  # Inductance in Henry
C = 10e-6  # Capacitance in Farad
target_f = 1000  # Target frequency in Hz
tolerance = 0.1  # Error tolerance in Ohm

# Function to calculate f(R)
def f_R(R):
    term_inside_sqrt = 1 / (L * C) - (R*2) / (4 * L*2)
    if term_inside_sqrt <= 0:
        return None  # Invalid if square root is negative
    return (1 / (2 * np.pi)) * np.sqrt(term_inside_sqrt)

# Derivative of f(R) for Newton-Raphson method
def f_prime_R(R):
    term_inside_sqrt = 1 / (L * C) - (R*2) / (4 * L*2)
    if term_inside_sqrt <= 0:
        return None  # Invalid if derivative is undefined
    sqrt_term = np.sqrt(term_inside_sqrt)
    return -R / (4 * np.pi * L**2 * sqrt_term)

# Newton-Raphson method implementation
def newton_raphson_method(initial_guess, tolerance):
    R = initial_guess
    while True:
        f_val = f_R(R)
        if f_val is None:
            return None  # Invalid case
        f_value = f_val - target_f
        f_prime_value = f_prime_R(R)
        if f_prime_value is None:
            return None  # Invalid case
        new_R = R - f_value / f_prime_value
        if abs(new_R - R) < tolerance:
            return new_R
        R = new_R

# Bisection method implementation
def bisection_method(a, b, tolerance):
    while (b - a) / 2 > tolerance:
        mid = (a + b) / 2
        f_mid = f_R(mid) - target_f
        if f_mid is None:
            return None  # Invalid case
        if abs(f_mid) < tolerance:
            return mid
        if (f_R(a) - target_f) * f_mid < 0:
            b = mid
        else:
            a = mid
    return (a + b) / 2

# Execute both methods
initial_guess = 50  # Initial guess for Newton-Raphson
interval_a, interval_b = 0, 100  # Bisection interval

# Results from Newton-Raphson
R_newton = newton_raphson_method(initial_guess, tolerance)
f_newton = f_R(R_newton) if R_newton is not None else "Not found"

# Results from Bisection method
R_bisection = bisection_method(interval_a, interval_b, tolerance)
f_bisection = f_R(R_bisection) if R_bisection is not None else "Not found"

# Display results
print("Newton-Raphson Method:")
print(f"Value of R: {R_newton} ohm, Resonant Frequency: {f_newton} Hz")

print("\nBisection Method:")
print(f"Value of R: {R_bisection} ohm, Resonant Frequency: {f_bisection} Hz")

# Plot results
plt.figure(figsize=(10, 5))
plt.axhline(target_f, color="red", linestyle="--", label="Target Frequency 1000 Hz")

# Plot Newton-Raphson results
if R_newton is not None:
    plt.scatter(R_newton, f_newton, color="blue", label="Newton-Raphson", zorder=5)
    plt.text(R_newton, f_newton + 30, f"NR: R={R_newton:.2f}, f={f_newton:.2f} Hz", color="blue")

# Plot Bisection results
if R_bisection is not None:
    plt.scatter(R_bisection, f_bisection, color="green", label="Bisection", zorder=5)
    plt.text(R_bisection, f_bisection + 30, f"Bisection: R={R_bisection:.2f}, f={f_bisection:.2f} Hz", color="green")

# Labeling the plot
plt.xlabel("Value of R (Ohm)")
plt.ylabel("Resonant Frequency f(R) (Hz)")
plt.title("Comparison of Newton-Raphson and Bisection Methods")
plt.legend()
plt.grid(True)
plt.show()


# Gaussian Elimination Method

import numpy as np

# Coefficient matrix and constant vector
A = np.array([[1, 1, 1],
              [1, 2, -1],
              [2, 1, 2]], dtype=float)

b = np.array([6, 2, 10], dtype=float)

# Gaussian elimination implementation
def gauss_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])  # Combine A and b

    # Elimination process
    for i in range(n):
        for j in range(i + 1, n):
            ratio = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= ratio * Ab[i, i:]

    # Back substitution process
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]

    return x

# Gaussian-Jordan elimination implementation
def gauss_jordan(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])  # Combine A and b

    # Elimination process
    for i in range(n):
        Ab[i] = Ab[i] / Ab[i, i]  # Make diagonal element 1
        for j in range(n):
            if i != j:
                Ab[j] -= Ab[i] * Ab[j, i]

    return Ab[:, -1]  # Solution

# Running both Gaussian methods
solution_gauss = gauss_elimination(A, b)
solution_gauss_jordan = gauss_jordan(A, b)

# Displaying results
print("Solution using Gaussian Elimination:")
print(f"x1 = {solution_gauss[0]}, x2 = {solution_gauss[1]}, x3 = {solution_gauss[2]}")

print("\nSolution using Gaussian-Jordan Elimination:")
print(f"x1 = {solution_gauss_jordan[0]}, x2 = {solution_gauss_jordan[1]}, x3 = {solution_gauss_jordan[2]}")

# Comparison of Errors for Different Numerical Methods

import numpy as np

# Function to compute R(T)
def R(T):
    return 5000 * np.exp(3500 * (1/T - 1/298))

# Numerical differentiation methods

# Forward difference method
def forward_difference(T, h):
    return (R(T + h) - R(T)) / h

# Backward difference method
def backward_difference(T, h):
    return (R(T) - R(T - h)) / h

# Central difference method
def central_difference(T, h):
    return (R(T + h) - R(T - h)) / (2 * h)

# Exact derivative calculation
def exact_derivative(T):
    return 5000 * np.exp(3500 * (1/T - 1/298)) * (-3500 / T**2)

# Temperature range and interval
temperatures = np.arange(250, 351, 10)
h = 1e-3  # Small interval for differences

# Storing results for each method
results = {
    "Temperature (K)": temperatures,
    "Forward Difference": [forward_difference(T, h) for T in temperatures],
    "Backward Difference": [backward_difference(T, h) for T in temperatures],
    "Central Difference": [central_difference(T, h) for T in temperatures],
    "Exact Derivative": [exact_derivative(T) for T in temperatures],
}

import matplotlib.pyplot as plt

# Calculating relative errors
errors = {
    "Forward Difference Error": np.abs((np.array(results["Forward Difference"]) - np.array(results["Exact Derivative"])) / np.array(results["Exact Derivative"])) * 100,
    "Backward Difference Error": np.abs((np.array(results["Backward Difference"]) - np.array(results["Exact Derivative"])) / np.array(results["Exact Derivative"])) * 100,
    "Central Difference Error": np.abs((np.array(results["Central Difference"]) - np.array(results["Exact Derivative"])) / np.array(results["Exact Derivative"])) * 100,
}

# Plotting relative errors
plt.figure(figsize=(10, 6))
plt.plot(temperatures, errors["Forward Difference Error"], label="Forward Difference Error", marker='o')
plt.plot(temperatures, errors["Backward Difference Error"], label="Backward Difference Error", marker='s')
plt.plot(temperatures, errors["Central Difference Error"], label="Central Difference Error", marker='^')
plt.xlabel("Temperature (K)")
plt.ylabel("Relative Error (%)")
plt.legend()
plt.title("Relative Error of Numerical Derivatives vs. Exact Derivative")
plt.grid()
plt.show()

