# app/services/methods.py

from sympy import * 
import numpy as np
import json

def biseccion(eqn, xi, xf, tol):
    eqn = sympify(eqn)  
    f = lambdify('x', eqn, "numpy") 
    
    iteraciones = [] 
    niter = 100000
    xm = 0
    cont = 0
    err = abs(xi - xf)

   
    if f(xi) * f(xf) == 0:
        if f(xi) == 0:
            return {"raiz": xi, "mensaje": "Xi es raíz"}
        else:
            return {"raiz": xf, "mensaje": "Xf es raíz"}
    elif f(xi) * f(xf) > 0:
        return {"raiz": None, "mensaje": "No se puede asegurar una raíz"}

 
    while err > tol and niter > cont:
        xm = (xi + xf) / 2
        f_xm = f(xm)

        iteraciones.append({
            "iteracion": cont + 1,
            "a": xi,
            "b": xf,
            "xm": xm,
            "f_xm": f_xm,
            "error": err
        })
        
        if f(xi) * f_xm < 0:
            xf = xm
        else:
            xi = xm
        
        err = abs(xi - xf)
        cont += 1

    # Final result
    if abs(f(xm)) < tol:
        return {
            "iteraciones": iteraciones,
            "raiz": xm,
            "mensaje": "Raíz encontrada con la tolerancia especificada."
        }
    elif err < tol:
        return {
            "iteraciones": iteraciones,
            "raiz": xm,
            "mensaje": "Xm es raíz con tolerancia: " + str(xm)
        }
    else:
        return {
            "iteraciones": iteraciones,
            "raiz": None,
            "mensaje": "No se obtuvo solución"
        }

def raicesMultiples(eqn_,eqn1_,eqn2_,xo,tol):

    eqn=sympify(eqn_)
    f = lambda x:eqn.subs({'x':x})
    eqn1=sympify(eqn1_)
    _f = lambda x:eqn1.subs({'x':x})
    eqn2=sympify(eqn2_)
    __f = lambda x:eqn2.subs({'x':x})
    
    niter = 100000
    cont = 0
    err = tol + 1

    while((err > tol) and (niter > cont)):
        fxo = f(xo)
        _fxo = _f(xo)
        __fxo = __f(xo)
        xn = xo - (fxo*_fxo)/((_fxo**2)-(fxo*__fxo))
        err = abs(xn-xo)
        xo = xn
        cont = cont + 1

    if(err <= tol):
        return {
            "raiz": str(xo),  
            "iteraciones": str(cont),
            "mensaje": "Raíz encontrada con la tolerancia especificada."
            
        }

        #return("se encontró una raíz en: " + str(xo) + " en: " +str(cont) + " iteraciones")
    else:
        return("El método no logró converger")
    
def reglaFalsa(eqn_, xi, xf, tol_):
    eqn = sympify(eqn_)  
    f = lambdify('x', eqn, "numpy")  
    iteraciones = []  
    niter = 100000
    xr = 0
    cont = 0
    err = abs(xi - xf)

 
    if f(xi) * f(xf) == 0:
        if f(xi) == 0:
            return {"raiz": xi, "mensaje": "Xi es raíz"}
        else:
            return {"raiz": xf, "mensaje": "Xf es raíz"}
    elif f(xi) * f(xf) > 0:
        return {"raiz": None, "mensaje": "No se puede asegurar una raíz"}

  
    while err > tol_ and niter > cont:
        xr = xf - (f(xf) * (xf - xi)) / (f(xf) - f(xi))  
        f_xr = f(xr)  
        
       
        iteraciones.append({
            "iteracion": cont + 1,
            "a": xi,
            "b": xf,
            "xr": xr,
            "f_xr": f_xr,
            "error": err
        })
        
        if f(xi) * f_xr < 0:
            xf = xr
        else:
            xi = xr
        
        err = abs(xr - xi)  # Calculate error
        cont += 1

    # Final result
    if abs(f(xr)) < tol_:
        return {
            "iteraciones": iteraciones,
            "raiz": xr,
            "mensaje": "Raíz encontrada con la tolerancia especificada."
        }
    elif err < tol_:
        return {
            "iteraciones": iteraciones,
            "raiz": xr,
            "mensaje": "Xm es raíz con tolerancia: " + str(xr)
        }
    else:
        return {
            "iteraciones": iteraciones,
            "raiz": None,
            "mensaje": "No se obtuvo solución"
        }

def newton(eqn_, eqn1_, xo, tol):
    eqn = sympify(eqn_)  
    f = lambdify('x', eqn, "numpy")  
    
    eqn1 = sympify(eqn1_) 
    g = lambdify('x', eqn1, "numpy") 
    
    niter = 100000
    cont = 0
    err = tol + 1
    iteraciones = [] 
    
    while err > tol and cont < niter:
        fxo = f(xo)
        gxo = g(xo)
        
        if gxo == 0:  
            return {
                "iteraciones": iteraciones,
                "raiz": None,
                "mensaje": "Derivada igual a cero en: " + str(cont) + " iteraciones"
            }
        
        xn = xo - fxo / gxo 
        f_x0 = fxo
        f_prime_x0 = gxo
        x1 = xn
        error = abs(x1 - xo)
        
        iteraciones.append({
            "iteracion": cont + 1,
            "x0": xo,
            "f_x0": f_x0,
            "f_prime_x0": f_prime_x0,
            "x1": x1,
            "error": error
        })
        
        xo = xn  
        err = error
        cont += 1 

    # Final result
    if err <= tol:
        return {
            "iteraciones": iteraciones,
            "raiz": xn,
            "mensaje": "Raíz encontrada con la tolerancia especificada."
        }
    else:
        return {
            "iteraciones": iteraciones,
            "raiz": None,
            "mensaje": "El método no logró converger"
        }

def puntoFijo(eqn_,eqn2_,valorA_,tol_):

    eqn = sympify(eqn_)
    fx = lambda x:eqn.subs({'x':x})
    
    eqn2 = sympify(eqn2_)
    gx = lambda x:eqn2.subs({'x':x})
    
    a = valorA_
    tolera = tol_
    iteramax = 100000
    i = 0 
    b = gx(a)
    tramo = abs(b-a)

    while(tramo>=tolera and i<=iteramax):
        a = b
        b = gx(a)
        tramo = abs(b-a)
        i = i + 1
    respuesta = b
    
    if (i>=iteramax ):
        respuesta = np.nan
        
    return("la raiz es: " + str(respuesta), "Error: ", tramo)

def splineLineal(x, y):
    trazadores = []
    n = len(x)
    
    for i in range(n - 1):
     
        m = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) 
        b = y[i] - m * x[i] 
        trazadores.append(f"{m:.6f}x + {b:.6f}")
    
    return trazadores

def sustitucion_progresiva(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i] 
    return y

def sustitucion_regresiva(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x

def crout(A,b):
    Anp = np.array(A)
    bnp = np.array(b)
    n = Anp.shape[0]
    L = np.zeros_like(Anp)
    U = np.eye(n)  

    for j in range(n):
        for i in range(j, n):
            L[i][j] = Anp[i][j] - sum(L[i][k] * U[k][j] for k in range(j))
        for i in range(j + 1, n):
            U[j][i] = (Anp[j][i] - sum(L[j][k] * U[k][i] for k in range(j))) / L[j][j]

    y = sustitucion_progresiva(L, bnp)

    x = sustitucion_regresiva(U, y)

    return {"L": L.tolist(), "U": U.tolist(), "x": x.tolist()}

def dolittle(A,b):

    Anp = np.array(A)
    bnp = np.array(b)
    n = Anp.shape[0]
    L = np.eye(n)  
    U = np.zeros_like(Anp)

    for i in range(n):
        for j in range(i, n):
            U[i][j] = Anp[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i + 1, n):
            L[j][i] = (Anp[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    y = sustitucion_progresiva(L, bnp)
    x = sustitucion_regresiva(U, y)

    return {"L": L.tolist(), "U": U.tolist(), "x": x.tolist()}

def luSimple(A,b):

    Anp = np.array(A)
    bnp = np.array(b)
    n = Anp.shape[0]
    L = np.zeros_like(Anp)
    U = np.zeros_like(Anp)
    
    for i in range(n):
        L[i][i] = 1
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i + 1, n):
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
    
    y = sustitucion_progresiva(L, bnp)

    x = sustitucion_regresiva(U, y)
    return {"L": L.tolist(), "U": U.tolist(), "x": x.tolist()}

def luPivoteo(A,b):
    Anp = np.array(A)
    bnp = np.array(b)
    n = Anp.shape[0]
    P = np.eye(n)  # Matriz de permutación
    L = np.zeros_like(Anp)
    U = Anp.copy()

    for i in range(n):

        max_index = np.argmax(np.abs(U[i:, i])) + i
        if max_index != i:
          
            U[[i, max_index], :] = U[[max_index, i], :]
            P[[i, max_index], :] = P[[max_index, i], :]

            L[[i, max_index], :i] = L[[max_index, i], :i]
        
        L[i][i] = 1
        for j in range(i + 1, n):
            L[j][i] = U[j][i] / U[i][i]
            U[j] -= L[j][i] * U[i]
    
    y = sustitucion_progresiva(L, np.dot(P, bnp))  # Usar P*b para b permutado
    x = sustitucion_regresiva(U, y)
    
    return {"L": L.tolist(), "U": U.tolist(),"P:": P.tolist() , "x": x.tolist()}

def polinomio_Newton(x_valor, y_valor):
    """
    Calcula el polinomio interpolador de Newton para un conjunto de puntos dados.

    Args:
        x_valor (list[float]): Lista de valores de x.
        y_valor (list[float]): Lista de valores de y correspondientes a x.

    Returns:
        str: El polinomio de Newton en forma simbólica como cadena.
    """
    # Crear la variable simbólica x
    x = Symbol('x')
    
    # Número de puntos
    n = len(x_valor)
    
    # Matriz de diferencias divididas
    diferencias_divididas = [[0 for _ in range(n)] for _ in range(n)]
    
    # Inicializar la primera columna con los valores de y
    for i in range(n):
        diferencias_divididas[i][0] = y_valor[i]
    
    # Calcular las diferencias divididas
    for j in range(1, n):
        for i in range(n - j):
            denominador = (x_valor[i + j] - x_valor[i])
            if denominador == 0:
                raise ValueError(
                    f"División por cero al calcular diferencias divididas: x[{i + j}] y x[{i}] son iguales"
                )
            diferencias_divididas[i][j] = (
                diferencias_divididas[i + 1][j - 1] - diferencias_divididas[i][j - 1]
            ) / denominador
    
    # Construir el polinomio de Newton
    polinomio = diferencias_divididas[0][0]
    termino = 1  # Acumula los términos del polinomio
    for i in range(1, n):
        termino *= (x - x_valor[i - 1])
        polinomio += diferencias_divididas[0][i] * termino
    
    # Simplificar el polinomio
    polinomio = sp.simplify(polinomio)
    
    return polinomio

def lagrange(x_vals, y_vals):
    """
    Calcula el polinomio de Lagrange dados puntos x_vals y y_vals.
    """
    x = sp.Symbol('x')
    n = len(x_vals)

    # Inicializa el polinomio como 0
    polinomio = 0

    # Construcción del polinomio de Lagrange
    for i in range(n):
        L_i = 1
        for j in range(n):
            if i != j:
                L_i *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
        polinomio += y_vals[i] * L_i

    # Simplificar el polinomio resultante
    polinomio = simplify(polinomio)
    return str(polinomio)

def jacobi_method(matrix_a, vector_b, x0, tol, niter):
    """
    Método iterativo de Jacobi para resolver sistemas de ecuaciones lineales.

    Args:
        matrix_a (list[list[float]]): Matriz de coeficientes.
        vector_b (list[float]): Vector de términos independientes.
        x0 (list[float]): Vector inicial.
        tol (float): Tolerancia.
        niter (int): Número máximo de iteraciones.

    Returns:
        dict: Resultado con información del método.
    """
    A = np.array(matrix_a)
    b = np.array(vector_b).reshape((-1, 1))
    x0 = np.array(x0).reshape((-1, 1))

    # Descomposición de A en D, L y U
    D = np.diag(np.diag(A))
    L = -1 * np.tril(A) + D
    U = -1 * np.triu(A) + D

    # Cálculo de T y C
    T = np.linalg.inv(D) @ (L + U)
    C = np.linalg.inv(D) @ b

    # Radio espectral
    spectral_radius = max(abs(np.linalg.eigvals(T)))
    converges = spectral_radius < 1

    iterations = []
    xP = x0
    for i in range(niter):
        xA = T @ xP + C
        error = np.linalg.norm(xP - xA)
        xP = xA

        iterations.append({"step": i + 1, "x": xA.flatten().tolist(), "error": error})
        if error < tol:
            break

    return {
        "transition_matrix": T.tolist(),
        "coefficient_matrix": C.tolist(),
        "spectral_radius": spectral_radius,
        "iterations": iterations,
        "converges": converges,
    }

def procesar_funcion(funcion_str):
    """
    Procesa una función en cadena para reemplazar operadores incompatibles.
    """
    funcion_str = funcion_str.replace("ln(", "log(")  # Cambiar ln a log para logaritmo natural
    funcion_str = funcion_str.replace("^", "**")      # Cambiar ^ a ** para potencias
    return funcion_str

def busquedas_incrementales(funcion_str, x0, intervalo, tol, max_iter):
    """
    Método de Búsquedas Incrementales para encontrar intervalos con raíces.

    Args:
        funcion_str (str): Función matemática como cadena.
        x0 (float): Valor inicial.
        intervalo (float): Tamaño del incremento.
        tol (float): Tolerancia.
        max_iter (int): Máximo de iteraciones.

    Returns:
        dict: Resultado con los intervalos y errores.
    """
    # Procesar la función y convertirla en evaluable
    funcion_str = procesar_funcion(funcion_str)
    f = lambda x: eval(funcion_str, {"x": x, "sin": sin, "cos": cos, "exp": exp, "log": log, "math": math})
    
    # Inicializar variables
    x_i = x0
    iteraciones = 0
    tabla = []

    # Evaluar la función inicial
    try:
        f_xi = f(x_i)
    except (ValueError, ZeroDivisionError) as e:
        return {"error": f"Error al evaluar f({x_i}): {str(e)}"}

    while iteraciones < max_iter:
        x_i_next = x_i + intervalo
        try:
            f_xi_next = f(x_i_next)
        except (ValueError, ZeroDivisionError) as e:
            return {"error": f"Error al evaluar f({x_i_next}): {str(e)}"}
        
        # Calcular el error absoluto
        error_abs = abs(x_i_next - x_i)
        tabla.append({
            "iteracion": iteraciones + 1,
            "x_i": x_i,
            "f_xi": f_xi,
            "x_i_next": x_i_next,
            "f_xi_next": f_xi_next,
            "error_abs": error_abs
        })

        # Verificar si hay cambio de signo
        if f_xi * f_xi_next < 0:
            return {
                "intervalo_raiz": [x_i, x_i_next],
                "iteraciones": tabla,
                "mensaje": f"Raíz aproximada en el intervalo [{x_i}, {x_i_next}]"
            }
        
        # Verificar tolerancia
        if error_abs < tol:
            return {
                "intervalo_raiz": [x_i, x_i_next],
                "iteraciones": tabla,
                "mensaje": f"El método alcanzó la tolerancia en la iteración {iteraciones + 1}."
            }

        # Avanzar al siguiente punto
        x_i = x_i_next
        f_xi = f_xi_next
        iteraciones += 1

    return {"mensaje": "No se encontraron raíces después del número máximo de iteraciones.", "iteraciones": tabla}

def eliminacion_gaussiana(A, b):
    """
    Método de Eliminación Gaussiana para resolver sistemas de ecuaciones lineales.

    Args:
        A (list[list[float]]): Matriz de coeficientes (cuadrada).
        b (list[float]): Vector de términos independientes.

    Returns:
        dict: Resultado con las soluciones o un mensaje de error.
    """
    # Convertir A y b a matrices de numpy
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)

    # Unión de A y b para formar la matriz aumentada
    matriz_aumentada = np.hstack((A, b))

    # Número de filas
    n = len(A)

    # Aplicación del método de eliminación gaussiana
    for i in range(n):
        # Verificar si el pivote es cero
        if matriz_aumentada[i, i] == 0:
            return {"error": f"No se puede realizar la eliminación: pivote igual a cero en la fila {i + 1}."}

        # Transformar en cero las entradas de la columna i en las filas debajo del pivote
        for j in range(i + 1, n):
            factor = matriz_aumentada[j, i] / matriz_aumentada[i, i]
            matriz_aumentada[j, i:] -= factor * matriz_aumentada[i, i:]

    # Sustitución regresiva para encontrar las soluciones
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (matriz_aumentada[i, -1] - np.dot(matriz_aumentada[i, i + 1:n], x[i + 1:n])) / matriz_aumentada[i, i]

    # Formatear las soluciones como una lista
    soluciones = [{"variable": f"x{i + 1}", "valor": x[i]} for i in range(n)]

    return {
        "matriz_aumentada": matriz_aumentada.tolist(),
        "soluciones": soluciones,
    }

def pivoteo_parcial(A, b):
    """
    Método de Pivoteo Parcial para resolver sistemas de ecuaciones lineales.

    Args:
        A (list[list[float]]): Matriz de coeficientes (cuadrada).
        b (list[float]): Vector de términos independientes.

    Returns:
        dict: Resultado con las soluciones o mensaje de error.
    """
    # Convertir A y b a matrices de numpy
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)

    # Unión de A y b para formar la matriz aumentada
    matriz_aumentada = np.hstack((A, b))

    # Número de filas
    n = len(A)

    # Aplicación del método de pivoteo parcial
    for i in range(n):
        # Seleccionar el valor absoluto más grande en la columna i, desde la fila i hacia abajo
        max_index = np.argmax(abs(matriz_aumentada[i:, i])) + i

        # Intercambiar la fila actual con la fila que tiene el valor absoluto mayor
        if max_index != i:
            matriz_aumentada[[i, max_index]] = matriz_aumentada[[max_index, i]]

        # Transformar en cero las entradas de la columna i en las filas debajo del pivote
        for j in range(i + 1, n):
            factor = matriz_aumentada[j, i] / matriz_aumentada[i, i]
            matriz_aumentada[j, i:] -= factor * matriz_aumentada[i, i:]

    # Sustitución regresiva para encontrar las soluciones
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (matriz_aumentada[i, -1] - np.dot(matriz_aumentada[i, i + 1:n], x[i + 1:n])) / matriz_aumentada[i, i]

    # Formatear las soluciones como una lista
    soluciones = [{"variable": f"x{i + 1}", "valor": x[i]} for i in range(n)]

    return {
        "matriz_aumentada": matriz_aumentada.tolist(),
        "soluciones": soluciones,
    }

def pivoteo_total(A, b):
    """
    Método de Pivoteo Total para resolver sistemas de ecuaciones lineales.

    Args:
        A (list[list[float]]): Matriz de coeficientes (cuadrada).
        b (list[float]): Vector de términos independientes.

    Returns:
        dict: Resultado con las soluciones o mensaje de error.
    """
    # Convertir A y b a matrices de numpy
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)

    # Unión de A y b para formar la matriz aumentada
    matriz_aumentada = np.hstack((A, b))

    # Vector de seguimiento para el reordenamiento de columnas
    orden_columnas = np.arange(len(A))

    # Número de filas
    n = len(A)

    # Aplicación del método de pivoteo total
    for i in range(n):
        # Buscar el índice del valor absoluto máximo en la submatriz restante
        submatriz = abs(matriz_aumentada[i:, i:n])
        max_index = np.unravel_index(np.argmax(submatriz, axis=None), submatriz.shape)
        max_fila = max_index[0] + i
        max_col = max_index[1] + i

        # Intercambio de filas
        if max_fila != i:
            matriz_aumentada[[i, max_fila]] = matriz_aumentada[[max_fila, i]]

        # Intercambio de columnas
        if max_col != i:
            matriz_aumentada[:, [i, max_col]] = matriz_aumentada[:, [max_col, i]]
            orden_columnas[[i, max_col]] = orden_columnas[[max_col, i]]

        # Transformar en cero las entradas de la columna i en las filas debajo del pivote
        for j in range(i + 1, n):
            factor = matriz_aumentada[j, i] / matriz_aumentada[i, i]
            matriz_aumentada[j, i:] -= factor * matriz_aumentada[i, i:]

    # Sustitución regresiva para encontrar las soluciones
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (matriz_aumentada[i, -1] - np.dot(matriz_aumentada[i, i + 1:n], x[i + 1:n])) / matriz_aumentada[i, i]

    # Reordenar las soluciones de acuerdo a los intercambios de columnas
    x_final = np.zeros(n)
    for i in range(n):
        x_final[orden_columnas[i]] = x[i]

    # Formatear las soluciones como una lista
    soluciones = [{"variable": f"x{i + 1}", "valor": x_final[i]} for i in range(n)]

    return {
        "matriz_aumentada": matriz_aumentada.tolist(),
        "soluciones": soluciones,
    }

def cholesky_method(matrix_a, vector_b):
    """
    Método de Descomposición de Cholesky para resolver sistemas de ecuaciones lineales.

    Args:
        matrix_a (list[list[float]]): Matriz de coeficientes (cuadrada y simétrica).
        vector_b (list[float]): Vector de términos independientes.

    Returns:
        dict: Resultado con la solución, matrices L y U, y etapas.
    """
    A = np.array(matrix_a, dtype=np.float64)
    b = np.array(vector_b).reshape((-1, 1))

    # Verificar que A sea simétrica
    if not np.allclose(A, A.T):
        raise ValueError("La matriz A debe ser simétrica para aplicar Cholesky.")

    # Descomposición de Cholesky manual
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    U = np.zeros((n, n), dtype=np.float64)
    etapas = []

    for k in range(n):
        # Calcular L[k][k]
        sum1 = sum(L[k][p] * U[p][k] for p in range(k))
        L[k][k] = np.sqrt(A[k][k] - sum1)
        U[k][k] = L[k][k]

        # Calcular L[i][k] para i > k
        for i in range(k + 1, n):
            sum2 = sum(L[i][r] * U[r][k] for r in range(k))
            L[i][k] = (A[i][k] - sum2) / U[k][k]

        # Calcular U[k][j] para j > k
        for j in range(k + 1, n):
            sum3 = sum(L[k][s] * U[s][j] for s in range(k))
            U[k][j] = (A[k][j] - sum3) / L[k][k]

        # Guardar la etapa actual
        etapas.append({
            "etapa": k + 1,
            "L": L.copy().tolist(),
            "U": U.copy().tolist()
        })

    # Resolviendo Ly = b (sustitución progresiva)
    y = np.zeros_like(b, dtype=np.float64)
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]

    # Resolviendo Ux = y (sustitución regresiva)
    x = np.zeros_like(b, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

    return {
        "solution": x.flatten().tolist(),
        "lower_triangular_matrix": L.tolist(),
        "upper_triangular_matrix": U.tolist(),
        "stages": etapas
    }

def gauss_seidel_method(matrix_a, vector_b, x0, tol, niter):
    """
    Método de Gauss-Seidel para resolver sistemas de ecuaciones lineales.

    Args:
        matrix_a (list[list[float]]): Matriz de coeficientes (cuadrada).
        vector_b (list[float]): Vector de términos independientes.
        x0 (list[float]): Vector inicial.
        tol (float): Tolerancia.
        niter (int): Número máximo de iteraciones.

    Returns:
        dict: Resultado con las iteraciones, matrices y convergencia.
    """
    A = np.array(matrix_a, dtype=float)
    b = np.array(vector_b, dtype=float).reshape((-1, 1))
    x0 = np.array(x0, dtype=float).reshape((-1, 1))

    # Descomposición de A en D, L y U
    D = np.diag(np.diag(A))
    L = -1 * np.tril(A) + D
    U = -1 * np.triu(A) + D

    # Cálculo de T y C
    T = np.linalg.inv(D - L) @ U
    C = np.linalg.inv(D - L) @ b

    # Radio espectral
    spectral_radius = max(abs(np.linalg.eigvals(T)))
    converges = spectral_radius < 1

    iterations = []
    xP = x0
    for i in range(niter):
        xA = T @ xP + C
        error = np.linalg.norm(xP - xA)
        xP = xA

        iterations.append({"step": i + 1, "x": xA.flatten().tolist(), "error": error})
        if error < tol:
            break

    return {
        "transition_matrix": T.tolist(),
        "coefficient_matrix": C.tolist(),
        "spectral_radius": spectral_radius,
        "iterations": iterations,
        "converges": converges,
    }

def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Método de la Secante para encontrar raíces de una función.

    Args:
        f (str): Función matemática como cadena.
        x0 (float): Primera estimación inicial.
        x1 (float): Segunda estimación inicial.
        tol (float): Tolerancia.
        max_iter (int): Número máximo de iteraciones.

    Returns:
        dict: Resultado con la raíz encontrada, el número de iteraciones, y los detalles de las iteraciones.
    """
    x = symbols('x')
    f_expr = sympify(f)
    f_lambdified = lambdify(x, f_expr)

    iterations = []

    for i in range(max_iter):
        # Evaluar la función en los puntos actuales
        f_x0 = f_lambdified(x0)
        f_x1 = f_lambdified(x1)

        # Verificar si f(x1) es suficientemente pequeño
        if abs(f_x1) < tol:
            iterations.append({"iteration": i, "x": x1, "f(x)": f_x1, "error": 0.0})
            return {
                "root": x1,
                "iterations": iterations,
                "converged": True,
                "message": f"Root found at x = {x1} with tolerance {tol}"
            }

        # Calcular el siguiente punto usando la fórmula de la secante
        if f_x1 == f_x0:
            raise ValueError("Division by zero in the Secant method")

        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        error = abs(x2 - x1)

        # Guardar los datos de la iteración
        iterations.append({"iteration": i, "x": x1, "f(x)": f_x1, "error": error})

        # Verificar si el error es menor que la tolerancia
        if error < tol:
            return {
                "root": x2,
                "iterations": iterations,
                "converged": True,
                "message": f"Root found at x = {x2} with tolerance {tol}"
            }

        # Actualizar los puntos para la siguiente iteración
        x0, x1 = x1, x2

    # Si no converge
    return {
        "root": None,
        "iterations": iterations,
        "converged": False,
        "message": "The Secant method did not converge within the maximum number of iterations"
    }

