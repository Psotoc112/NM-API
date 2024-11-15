# app/services/methods.py

from sympy import *
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
    


#def add(a,b):
#    result = a+b
#    return result