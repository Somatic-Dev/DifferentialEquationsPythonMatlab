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
