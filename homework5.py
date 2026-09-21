import numpy as np
import matplotlib.pyplot as plt

# HW5 2.1.39: dP/dt = (k + b cos 2*pi*t) P
# Solution: P(t) = P0 * exp(k t + (b / 2 pi) sin(2 pi t))
# Contrast with natural growth P' = kP (same k, same P0): P(t) = P0 e^(kt)

P0 = 100.0
k = 0.5
b = 3.0

def P_seasonal(t):
    return P0 * np.exp(k * t + (b / (2 * np.pi)) * np.sin(2 * np.pi * t))

def P_natural(t):
    return P0 * np.exp(k * t)

if __name__ == "__main__":
    plt.style.use("dark_background")
    t = np.linspace(0, 5, 1000)

    plt.figure(figsize=(10, 6))
    plt.plot(t, P_natural(t), color="#ff8c42", linewidth=1.6, linestyle="--",
             label="natural growth  P = P0 e^(kt)")
    plt.plot(t, P_seasonal(t), color="#00e5ff", linewidth=2.2,
             label="seasonal  P = P0 e^(kt + (b/2pi) sin 2pi t)")
    plt.xlabel("t (years)")
    plt.ylabel("P(t)")
    plt.title(f"HW5 2.1.39: seasonal vs natural growth  (P0={P0:.0f}, k={k}, b={b})")
    plt.legend()
    plt.grid(color="#555555", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("hw5_2_1_39.png", dpi=150)
    plt.close()
    print("wrote hw5_2_1_39.png")

def make_2_2_5():
    # HW5 2.2.5: dx/dt = x^2 - 4; critical points x = -2 (stable), x = 2 (unstable)
    from homework4 import slope_field, plot_solution, plot_extra_solutions

    def f(t, x):
        return x**2 - 4

    t_span = (0, 2)
    plt.figure(figsize=(10, 6))
    slope_field(f, x_range=t_span, y_range=(-4, 4), density=27)
    plot_extra_solutions(f, t_span, [(0, 0), (0, 3), (0, -3)])
    plot_solution(f, t_span, t0=0, x0=1, label="x(0) = 1")
    plt.axhline(-2, color="#8aff80", linestyle=":", linewidth=1.0)
    plt.axhline(2, color="#ff5555", linestyle=":", linewidth=1.0)
    plt.title("HW5 2.2.5: dx/dt = x^2 - 4  (x=-2 stable, x=+2 unstable)")
    plt.tight_layout()
    plt.savefig("hw5_2_2_5.png", dpi=150)
    plt.close()
