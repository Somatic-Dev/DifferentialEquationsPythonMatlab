import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def slope_field(f, x_range, y_range, density=25):
    """
    Generate a slope field for the differential equation dy/dx = f(x, y).

    Parameters:
        f: function - The function f(x, y) representing dy/dx.
        x_range: tuple - The range of x values (xmin, xmax).
        y_range: tuple - The range of y values (ymin, ymax).
        density: int - The density of the slope field (number of arrows per axis).
    """
    x = np.linspace(x_range[0], x_range[1], density)
    y = np.linspace(y_range[0], y_range[1], density)
    X, Y = np.meshgrid(x, y)

    # Compute slopes
    U = 1  # dx is always 1 for slope field
    V = f(X, Y)  # dy = f(x, y)

    # Normalize arrows for better visualization
    N = np.sqrt(U**2 + V**2)
    U, V = U / N, V / N

    plt.quiver(X, Y, U, V, angles="xy", color="#9aa4b2")
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.grid(color="#555555", linestyle="--", linewidth=0.5)

def plot_solution(f, t_span, t0, x0, num_points=500,
                  color="#00e5ff", linewidth=2.2, label=None, bound=1e3):
    """
    Plot the solution of the differential equation dx/dt = f(t, x) with initial condition x(t0) = x0.

    Parameters:
        f: function - The function f(x, y) representing dy/dx.
        t_span: tuple - The plotting range of t values (tmin, tmax).
        t0: float - The initial-condition time.
        x0: float - The initial condition x(t0) = x0.
        num_points: int - The number of points to compute the solution.
        color, linewidth, label - Curve styling; label defaults to "x(t0)=x0".
        bound: float - Stop integrating if |x| exceeds this (finite-time blowup guard).
    """
    def ode_system(t, y):
        return [f(t, y[0])]

    def blowup(t, y):
        return abs(y[0]) - bound
    blowup.terminal = True

    if label is None:
        label = f"x({t0}) = {x0}"

    t_min, t_max = t_span
    t_segments = []
    x_segments = []

    if t0 > t_min:
        left_n = max(2, num_points // 2)
        left_sol = solve_ivp(
            ode_system,
            (t0, t_min),
            [x0],
            t_eval=np.linspace(t0, t_min, left_n),
            events=blowup,
        )
        t_segments.append(left_sol.t[::-1])
        x_segments.append(left_sol.y[0][::-1])

    if t0 < t_max:
        right_n = max(2, num_points // 2 + 1)
        right_sol = solve_ivp(
            ode_system,
            (t0, t_max),
            [x0],
            t_eval=np.linspace(t0, t_max, right_n),
            events=blowup,
        )
        if t_segments:
            t_segments.append(right_sol.t[1:])
            x_segments.append(right_sol.y[0][1:])
        else:
            t_segments.append(right_sol.t)
            x_segments.append(right_sol.y[0])

    if not t_segments:
        t = np.array([t0], dtype=float)
        x = np.array([x0], dtype=float)
    else:
        t = np.concatenate(t_segments)
        x = np.concatenate(x_segments)

    plt.plot(t, x, color=color, linewidth=linewidth, label=label)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.legend()
    plt.grid(color="#555555", linestyle="--", linewidth=0.5)

# Muted colors for the non-assigned "several solutions" curves
EXTRA_COLORS = ["#ff8c42", "#8aff80", "#d78cff"]

def plot_extra_solutions(f, t_span, initial_conditions):
    """Thin, muted companion curves so the assigned (highlighted) one stands out."""
    for (t0, x0), color in zip(initial_conditions, EXTRA_COLORS):
        plot_solution(f, t_span, t0, x0,
                      color=color, linewidth=1.3, label=f"x({t0}) = {x0}")

def make_2_1_3():
    # 2.1.3: dx/dt = 1 - x^2, assigned x(0) = 3, plus other solutions
    def f(t, x):
        return 1 - x**2

    t_span = (-2, 2)
    plt.figure(figsize=(10, 6))
    slope_field(f, x_range=t_span, y_range=(-4, 4), density=27)
    plot_extra_solutions(f, t_span, [(0, 0), (0, 2)])
    plot_solution(f, t_span, t0=0, x0=3, label="assigned: x(0) = 3")
    plt.title("HW4 2.1.3: dx/dt = 1 - x^2  (assigned solution highlighted)")
    plt.tight_layout()
    plt.savefig("hw4_2_1_3.png", dpi=150)
    plt.close()

def make_2_1_6():
    # 2.1.6: dx/dt = 3x(x - 5), assigned x(0) = 2, plus other solutions
    def f(t, x):
        return 3 * x * (x - 5)

    t_span = (-1, 1)
    plt.figure(figsize=(10, 6))
    slope_field(f, x_range=t_span, y_range=(-2, 7), density=27)
    plot_extra_solutions(f, t_span, [(0, 1), (0, 4)])
    plot_solution(f, t_span, t0=0, x0=2, label="assigned: x(0) = 2")
    plt.title("HW4 2.1.6: dx/dt = 3x(x - 5)  (assigned solution highlighted)")
    plt.tight_layout()
    plt.savefig("hw4_2_1_6.png", dpi=150)
    plt.close()

def make_custom_2():
    # Homework 4 Custom 2 (based on 2.1.23):
    # dx/dt = 0.8x - 0.004x^2, x(0) = 50
    def f(t, x):
        return 0.8 * x - 0.004 * x**2

    plt.figure(figsize=(10, 6))
    slope_field(f, x_range=(0, 2), y_range=(0, 150), density=27)
    plot_solution(f, t_span=(0, 2), t0=0, x0=50, label="Solution with x(0)=50")
    plt.axhline(100, color="#ffcc00", linestyle="--", linewidth=1.2, label="x = 100")
    plt.title("HW4 Custom 2: dx/dt = 0.8x - 0.004x^2, x(0) = 50")
    plt.tight_layout()
    plt.savefig("hw4_custom2.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    plt.style.use("dark_background")
    make_2_1_3()
    make_2_1_6()
    make_custom_2()
    print("wrote hw4_2_1_3.png, hw4_2_1_6.png, hw4_custom2.png")
