import numpy as np

def factorial(n):
    """
    Calcula el factorial de n.
    """
    if n <= 0:
        return 1
    else:
        return n*factorial(n-1)

def hermite_rodrigues(n):
    """
    Calcula el n-ésimo polinomio de Hermite utilizando la fórmula de Rodrigues.
    """
    coef = 1/np.sqrt(2**n * factorial(n))
    def func(x):
        return coef * np.exp(-x**2/2) * np.power(x, n)
    return func

def hermite_rodrigues_deriv(n):
    """
    Calcula la derivada del n-ésimo polinomio de Hermite utilizando la fórmula de Rodrigues.
    """
    return lambda x: n*hermite_rodrigues(n-1)(x) - x*hermite_rodrigues(n)(x)

def hermite_zeros_rodrigues(n, tol=1e-6, max_iter=100):
    """
    Calcula los ceros del n-ésimo polinomio de Hermite utilizando el método de Newton-Raphson.
    """
    poly = hermite_rodrigues(n)
    deriv = hermite_rodrigues_deriv(n)
    zeros = []
    for i in range(1, n+1):
        # Inicializar la estimación inicial del cero usando la raíz del polinomio anterior.
        if i == 1:
            x0 = np.sqrt(2)
        else:
            x0 = zeros[-1]
        # Calcular el cero utilizando el método de Newton-Raphson.
        for j in range(max_iter):
            fx = poly(x0)
            if abs(fx) < tol:
                break
            fpx = deriv(x0)
            x1 = x0 - fx/fpx
            if abs(x1 - x0) < tol:
                break
            x0 = x1
        zeros.append(x1)
    return zeros




    

