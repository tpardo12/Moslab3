import numpy as np


def f(vec): 
    w, x, y, z = vec
    return ((w-1)**2 + (x-2)**2 + (y-3)**2 + (z-4)**2 
            + 0.3*w*x + 0.2*x*y - 0.1*y*z)  #Funcion 

def grad(vec):  #Gradiente 
    w, x, y, z = vec
    return np.array([
        2*(w-1) + 0.3*x,
        2*(x-2) + 0.3*w + 0.2*y,
        2*(y-3) + 0.2*x - 0.1*z,
        2*(z-4) - 0.1*y
    ], dtype=float)

def hess(vec): #hessiana 
    
    return np.array([
        [2.0, 0.3, 0.0,  0.0],
        [0.3, 2.0, 0.2,  0.0],
        [0.0, 0.2, 2.0, -0.1],
        [0.0, 0.0, -0.1, 2.0]
    ], dtype=float)


def newton_nd(x0, tol=1e-10, maxit=50, alpha0=1.0, backtracking=True):
    x = np.array(x0, dtype=float)
    hist = [(0, x.copy(), f(x), np.linalg.norm(grad(x)))]  #historial para registrar las trayectorias 
    for k in range(1, maxit+1): #iteraciones (max 50)
        g = grad(x)
        if np.linalg.norm(g) < tol: #condicion de parada (ver documento)
            break
        H = hess(x)
        # Resuelve H p = -g
        p = np.linalg.solve(H, -g)

     
        alpha = alpha0
        if backtracking: #se reduce iterativamente alpha  keintras la nueva evaluacion  f(x + alpha*p) no satisfaga el criterio de descenso suficiente o hasta que alpha sea muy pequeña.
            fx = f(x)
            c = 1e-4
            while f(x + alpha*p) > fx + c*alpha*np.dot(g, p) and alpha > 1e-8:
                alpha *= 0.5

        x = x + alpha*p
        hist.append((k, x.copy(), f(x), np.linalg.norm(grad(x))))
        if np.linalg.norm(hist[-1][1] - hist[-2][1]) < tol:
            break
    return x, hist


x0 = (0.0, 5.0, -2.0, 8.0)        
xmin, hist = newton_nd(x0)

print("Mínimo aproximado:", xmin)
print("f(min) =", f(xmin))

evals = np.linalg.eigvalsh(hess(xmin))
print("Autovalores de H en el mínimo:", evals)  


for it, xk, fk, ng in hist[:10]:
    print(f"iter {it:02d}: x={xk}, f={fk:.6e}, ||grad||={ng:.2e}")
