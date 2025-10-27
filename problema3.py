import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def f(xy):
    x, y = xy
    return (x-1)**2 + 100*(y - x**2)**2

def grad(xy):
    x, y = xy
    return np.array([
        2*(x-1) - 400*x*(y - x**2),
        200*(y - x**2)
    ], dtype=float)

def hess(xy):
    x, y = xy
    return np.array([
        [1200*x**2 - 400*y + 2,   -400*x],
        [-400*x,                   200  ]
    ], dtype=float)

def newton_nd(x0, alpha=1.0, tol=1e-8, maxit=100):
    x = np.array(x0, dtype=float)
    hist = [x.copy()]
    for _ in range(maxit):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        H = hess(x)
        p = np.linalg.solve(H, -g)     # resuelve H p = -g
        x_new = x + alpha*p
        hist.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x, np.array(hist)

def grafica_superficie(trayectoria, xmin, ymin, xmax, ymax, salida):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = np.linspace(xmin, xmax, 80)
    Y = np.linspace(ymin, ymax, 80)
    XX, YY = np.meshgrid(X, Y)
    ZZ = (XX-1)**2 + 100*(YY - XX**2)**2

    ax.plot_surface(XX, YY, ZZ, cmap="viridis", alpha=0.6, linewidth=0, antialiased=True)

    xs, ys = trayectoria[:,0], trayectoria[:,1]
    zs = np.array([f(p) for p in trayectoria])
    ax.plot(xs, ys, zs, color="black", marker="o", label="Iteraciones")
    ax.scatter(xs[-1], ys[-1], zs[-1], color="red", s=60, label="Mínimo final", zorder=5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.set_title("Superficie de f(x,y) con trayectoria de Newton")
    ax.legend()
    ax.view_init(elev=30, azim=-60)
    fig.tight_layout()
    fig.savefig(salida, dpi=160)

def clasifica_extremo(xy, tol=1e-10):
    """Determina el tipo de punto crítico a partir del Hessiano."""
    eigenvals = np.linalg.eigvalsh(hess(xy))
    if np.all(eigenvals > tol):
        tipo = "mínimo local"
    elif np.all(eigenvals < -tol):
        tipo = "máximo local"
    else:
        tipo = "punto de silla"
    return tipo, eigenvals

# Ejemplo con (x0,y0)=(0,10)
xmin, tray = newton_nd((0,10), alpha=1.0)
print("mínimo aproximado:", xmin, "f=", f(xmin))

print("Iteraciones del método de Newton:")
for k, punto in enumerate(tray):
    print(f" iter {k:02d}: x={punto[0]: .6f}, y={punto[1]: .6f}, f={f(punto): .6e}")

tipo_extremo, eigenvals = clasifica_extremo(xmin)
print(f"Clasificación del punto final: {tipo_extremo} (autovalores: {eigenvals})")

x_vals, y_vals = tray[:,0], tray[:,1]
margin_x = max(1.0, 0.2*(x_vals.max() - x_vals.min() + 1e-9))
margin_y = max(1.0, 0.2*(y_vals.max() - y_vals.min() + 1e-9))
xmin_plot, xmax_plot = x_vals.min() - margin_x, x_vals.max() + margin_x
ymin_plot, ymax_plot = y_vals.min() - margin_y, y_vals.max() + margin_y
out_path = Path(__file__).with_name("grafico_problema3.png")
grafica_superficie(tray, xmin_plot, ymin_plot, xmax_plot, ymax_plot, out_path)
