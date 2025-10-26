import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def f(x):  return 3*x**3 - 10*x**2 - 56*x + 50
def f1(x): return 9*x**2 - 20*x - 56
def f2(x): return 18*x - 20

def newton_step(x, alpha=1.0):
    return x - alpha * (f1(x) / f2(x))

def newton_sequence(x0, alpha=1.0, tol=1e-8, maxit=50):
    xs = [float(x0)]
    for _ in range(maxit):
        if abs(f2(xs[-1])) < 1e-12: 
            break
        x_next = newton_step(xs[-1], alpha)
        xs.append(x_next)
        
        if abs(xs[-1] - xs[-2]) < tol or abs(f1(xs[-1])) < tol:
            break
    return xs 

def clasifica(x_star):
    tipo = "mínimo" if f2(x_star) > 0 else "máximo" if f2(x_star) < 0 else "punto de inflexión"
    return {"x*": x_star, "f(x*)": f(x_star), "tipo": tipo }



critical_points = []

def guarda_punto(info, puntos, tol=1e-8):
    for p in puntos:
        if abs(p["x*"] - info["x*"]) < tol:
            return puntos
    puntos.append(info)
    return puntos

for x0 in [-6, -3, -2, -1.9, -1.8, -1.7 , 1.7 , 1.8 , 2.9, 3, 6]:
    alpha = 1.0
    xs = newton_sequence(x0, alpha)
    x_star = xs[-1]
    info = clasifica(x_star)
    critical_points = guarda_punto(info, critical_points)
    print(f"x0={x0:8.4f} → {len(xs)-1} iter, x*={x_star:.6f}, f(x*)={info['f(x*)']:.5f}, {info['tipo']} , aplha= {alpha} ")

for x0 in [-6, -5, -4, -3, -2, -1 , 1 , 2 , 3, 4, 5]:
    alpha = 0.6
    xs = newton_sequence(x0, alpha)
    x_star = xs[-1]
    info = clasifica(x_star)
    critical_points = guarda_punto(info, critical_points)
    print(f"x0={x0:8.4f} → {len(xs)-1} iter, x*={x_star:.6f}, f(x*)={info['f(x*)']:.5f}, {info['tipo']} ,{alpha} ")


# Gráfico de f(x)=3x^3-10x^2-56x+50 en [-6,6] con extremos marcados


# Mostrar valores resumidos
for p in critical_points:
    print(f"{p['tipo']} en x* ≈ {p['x*']:.6f}, f(x*) ≈ {p['f(x*)']:.6f}")

xs = np.linspace(-6, 6, 600)
ys = f(xs)

# Curva y puntos críticos provenientes del método
colores = {"máximo": "tab:red", "mínimo": "tab:green", "punto de inflexión": "tab:gray"}
plt.figure()
plt.plot(xs, ys, label="f(x)")

for p in critical_points:
    plt.scatter(p["x*"], p["f(x*)"], color=colores[p["tipo"]], zorder=3, label=p["tipo"])
    plt.annotate(
        p["tipo"],
        (p["x*"], p["f(x*)"]),
        xytext=(p["x*"] + (0.5 if p["tipo"] == "mínimo" else -1.5), p["f(x*)"] + 30),
        arrowprops=dict(arrowstyle="->")
    )

plt.title("f(x)=3x^3-10x^2-56x+50 con puntos críticos (Newton-Raphson)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
handles, labels = plt.gca().get_legend_handles_labels()
# Evitar etiquetas duplicadas si hay varios máximos/mínimos en la gráfica
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys())
plt.tight_layout()

# Guardar imagen en el mismo directorio del script
out_path = Path(__file__).with_name("grafico_problema1.png")
plt.savefig(out_path, dpi=160)
plt.show()

out_path
