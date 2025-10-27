import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def f(x):  return x**5 - 8*x**3 + 10*x + 6
def f1(x): return 5*x**4 - 24*x**2 + 10
def f2(x): return 20*x**3 - 48*x

def newton_step(x, alpha):
    return x - alpha * (f1(x) / f2(x))

def newton_sequence(x0, alpha, tol=1e-8, maxit=50):
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

def etiqueta_extremos_globales(puntos, tol=1e-8):
    """Marca qué máximos/mínimos son globales dentro de los puntos críticos."""
    for p in puntos:
        p["es_global"] = False

    maximos = [p for p in puntos if p["tipo"] == "máximo"]
    if maximos:
        valor_max = max(p["f(x*)"] for p in maximos)
        for p in maximos:
            if abs(p["f(x*)"] - valor_max) < tol:
                p["es_global"] = True

    minimos = [p for p in puntos if p["tipo"] == "mínimo"]
    if minimos:
        valor_min = min(p["f(x*)"] for p in minimos)
        for p in minimos:
            if abs(p["f(x*)"] - valor_min) < tol:
                p["es_global"] = True

def guarda_punto(info, puntos, tol=1e-8):
    for p in puntos:
        if abs(p["x*"] - info["x*"]) < tol:
            return puntos
    puntos.append(info)
    return puntos

for x0 in [-3,-2,-1,1,2,3]:
    alpha = 1.0
    xs = newton_sequence(x0, alpha)
    x_star = xs[-1]
    info = clasifica(x_star)
    critical_points = guarda_punto(info, critical_points)
    print(f"x0={x0:8.4f} → {len(xs)-1} iter, x*={x_star:.6f}, f(x*)={info['f(x*)']:.5f}, {info['tipo']} , aplha= {alpha} ")

for x0 in [-3,-2,-1,1,2,3]:
    alpha = 0.6
    xs = newton_sequence(x0, alpha)
    x_star = xs[-1]
    info = clasifica(x_star)
    critical_points = guarda_punto(info, critical_points)
    print(f"x0={x0:8.4f} → {len(xs)-1} iter, x*={x_star:.6f}, f(x*)={info['f(x*)']:.5f}, {info['tipo']} ,{alpha} ")

etiqueta_extremos_globales(critical_points)

# Gráfico de f(x)=3x^3-10x^2-56x+50 en [-6,6] con extremos marcados


# Mostrar valores resumidos
for p in critical_points:
    print(f"{p['tipo']} en x* ≈ {p['x*']:.6f}, f(x*) ≈ {p['f(x*)']:.6f}")

xs = np.linspace(-3.3, 3.3, 50)
ys = f(xs)

# Curva y puntos críticos provenientes del método
colores = {"punto de inflexión": "tab:gray"}
plt.figure()
plt.plot(xs, ys, label="f(x)")

for p in critical_points:
    if p["tipo"] == "punto de inflexión":
        color = colores[p["tipo"]]
    else:
        color = "tab:red" if p["es_global"] else "black"
    etiqueta = p["tipo"]
    if p["tipo"] in ("máximo", "mínimo"):
        etiqueta += " global" if p["es_global"] else " local"
    plt.scatter(p["x*"], p["f(x*)"], color=color, zorder=3, label=etiqueta)
    plt.annotate(
        etiqueta,
        (p["x*"], p["f(x*)"]),
        xytext=(p["x*"] + (0.5 if p["tipo"] == "mínimo" else -1.5), p["f(x*)"] + 30),
        arrowprops=dict(arrowstyle="->")
    )

plt.title(" con puntos críticos (Newton-Raphson)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
handles, labels = plt.gca().get_legend_handles_labels()
# Evitar etiquetas duplicadas si hay varios máximos/mínimos en la gráfica
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys())
plt.tight_layout()

# Guardar imagen en el mismo directorio del script
out_path = Path(__file__).with_name("grafico_problema2.png")
plt.savefig(out_path, dpi=160)
plt.show()

out_path
