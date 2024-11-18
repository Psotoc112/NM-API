# app/services/methods.py

from sympy import *
import numpy as np


def biseccion(eqn,xi,xf,tol):

    eqn = sympify(eqn)
    f = lambda x:eqn.subs({'x':x})
    niter = 100000
    xm = 0
    if(f(xi)*f(xf) == 0):
        if(f(xi) == 0):
            return("Xi es raíz")
        else:
            return("Xf es raíz")
    elif(f(xi)*f(xf) > 0):
        return("No se puede asegurar una raíz")
    else:
        xm = (xf+xi)/2
        cont = 0
        err = abs(xi - xf)
        while(err > tol and niter > cont and f(xm) != 0):
            if(f(xi)*f(xm) < 0):
                xf = xm
            else:
                xi = xm
            xm = (xf+xi)/2
            err = abs(xm - xi)
            cont= cont + 1
        if(f(xm) == 0):
           return("xm es raíz: " + str(xm) +"\n Iteraciones realizadas: " + str(cont))
        elif(err < tol):
            return("xm es raíz con tolerancia: " + str(tol) + " con xm: " + str(xm) + " con " + str(cont) + " iteraciones realizadas" )
        else:
            return("No se obtuvo solución")

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
        return("se encontró una raíz en: " + str(xo) + " en: " +str(cont) + " iteraciones")
    else:
        return("El método no logró converger")
    
def reglaFalsa(eqn_,xi_,xf_,tol_):

    eqn = sympify(eqn_)
    f = lambda x:eqn.subs({'x':x})
    xi = xi_
    xf = xf_
    niter = 100000
    tol = tol_
    xm = 0
    
    if(f(xi)*f(xf) == 0):
        if(f(xi) == 0):
            return("Xi es raíz")
        else:
            return("Xf es raíz")
    elif(f(xi)*f(xf) > 0):
        return("No se puede asegurar una raíz")
    else:
        xm = xf - ( f(xf) * ( xf - xi ) ) / ( f(xf) - f(xi) )
        cont = 0
        err = abs(xi - xf)
        while(err > tol and niter > cont and f(xm) != 0):
            if(f(xi)*f(xm) < 0):
                xf = xm
            else:
                xi = xm
            xm = xf - ( f(xf) * ( xf - xi ) ) / ( f(xf) - f(xi) )
            err = abs(xm - xi)
            cont= cont + 1
        if(f(xm) == 0):
            return("xm es raíz: " + str(xm) +"\n Iteraciones realizadas: " + str(cont))
        elif(err < tol):
            return("xm es raíz con tolerancia: " + str(tol) + " con xm: " + str(xm) +"\nIteraciones realizadas: " + str(cont))
        else:
            return("No se obtuvo solución")

def newton(eqn_,eqn1_,xo,tol):    

    eqn = sympify(eqn_)
    f = lambda x: eqn.subs({'x': x})
    
    eqn1 = sympify(eqn1_)
    g = lambda x: eqn1.subs({'x': x})
    

    niter = 100000
    cont = 0
    err = tol + 1
    

    while (err > tol) and (niter > cont):
        fxo = f(xo)
        gxo = g(xo)
        
        if gxo == 0:
            return "Derivada igual a cero en: " + str(cont) + " iteraciones"
        
        xn = xo - fxo / gxo
        err = abs(xn - xo)
        xo = xn
        cont += 1
    

    if err <= tol:
        return "Se encontró una raíz en: " + str(xo) + " con: " + str(cont) + " iteraciones"
    else:
        return "El método no logró converger"
    

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