import numpy as np

def Factorial(n):
    
    factorial = 1
    
    for i in range(1,n+1):
        factorial *= i
        
    return factorial
    
def Factoriales(k):
    
    factoriales = np.array([])
    
    for i in range(0,k+1):
        factorial = Factorial(i)
        factoriales = np.append(factoriales,{"Número": i, "Factorial":factorial})
        
    return factoriales
    
print(Factoriales(20))
