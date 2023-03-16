import numpy as np

R = 1  # radio de la esfera
n = 100  # número de cuadrados en cada lado de la grilla

x = np.linspace(-R, R, n+1)
y = np.linspace(-R, R, n+1)
xx, yy = np.meshgrid(x, y)

volumen = 0

for i in range(n):
    for j in range(n):
        x1, x2 = xx[i, j], xx[i+1, j+1]
        y1, y2 = yy[i, j], yy[i+1, j+1]
        # coordenadas de los cuatro vertices del cuadrado pequeño
        vertices = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        # valores de la funcion en los cuatro vertices
        valores = [np.sqrt(max(0, R**2 - x**2 - y**2)) for x, y in vertices]
        # promedio de la funcion en los cuatro vertices
        promedio = sum(valores) / 4
        # area del cuadrado pequeno
        area = (x2 - x1) * (y2 - y1)
        # volumen del cuadrado pequeno
        volumen += promedio * area

print("Volumen de la semiesfera de radio R =", R, ":", volumen)
