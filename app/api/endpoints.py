# app/api/endpoints.py

from fastapi import APIRouter, HTTPException
from app.services.methods import *
from app.models.params import *

router = APIRouter()

@router.post("/biseccion")
def get_biseccion(params: biseccionParams):

    eqn = params.eqn
    xi = params.xi
    xf = params.xf
    tol = params.tol
    result = biseccion(eqn,xi,xf,tol)
    return {"result": result}

@router.post("/raicesMultiples")
def get_raicesMultiples(params: raicesMParams):

    eqn = params.eqn
    eqn1 = params.eqn1
    eqn2 = params.eqn2
    xo = params.xo
    tol = params.tol
    result = raicesMultiples(eqn,eqn1,eqn2,xo,tol)
    return {"result": result}

@router.post("/reglaFalsa")
def get_reglaFalsa(params: reglaFalsaParams):

    eqn = params.eqn
    xi = params.xi
    xf = params.xf
    tol = params.tol
    result = reglaFalsa(eqn,xi,xf,tol)
    return {"result": result}

@router.post("/newton")
def get_newton(params: newtonParams):

    eqn = params.eqn
    eqn1 = params.eqn1
    xo = params.xo
    tol = params.tol
    result = newton(eqn,eqn1,xo,tol)
    return {"result": result}

@router.post("/puntoFijo")
def get_puntoFijo(params: puntoFParams):

    eqn = params.eqn
    eqn2 = params.eqn2
    a = params.valorA
    tol = params.tol
    result = puntoFijo(eqn,eqn2,a,tol)
    return {"result": result}

@router.post("/splineLineal")
def get_splineLineal(params: spline1Params):
    x = params.x
    y = params.y
    result = splineLineal(x,y)
    return {"result": result}

@router.post("/crout")
def get_crout(params: croutParams):
    A = params.A
    b = params.b

    result = crout(A,b)
    return {"result": result}

@router.post("/dolittle")
def get_dolittle(params: dolittleParams):
    A = params.A
    b = params.b

    result = dolittle(A,b)
    return {"result": result}

@router.post("/LUsimple")
def get_LUsimple(params: LUParams):
    A = params.A
    b = params.b

    result = luSimple(A,b)
    return {"result": result}

@router.post("/LUpartial")
def get_LUpartial(params: LUParams):
    A = params.A
    b = params.b

    result = luPivoteo(A,b)
    return {"result": result}

@router.post("/Newton")
def get_newton(params: NewtonParams):
    x_valor = params.x_valor
    y_valor = params.y_valor
    
    # Llama al método para calcular el polinomio de Newton
    try:
        result = polinomio_Newton(x_valor, y_valor)
        return {"result": str(result)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/lagrange")
def get_lagrange(params: LagrangeParams):
    try:
        x_vals = params.x
        y_vals = params.y

        # Validar que x e y tengan la misma longitud
        if len(x_vals) != len(y_vals):
            raise HTTPException(status_code=400, detail="Las listas x e y deben tener la misma longitud.")

        # Calcular el polinomio de Lagrange
        result = lagrange(x_vals, y_vals)
        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/jacobi")
def get_jacobi(params: JacobiParams):
    try:
        # Extraer los parámetros
        matrix_a = params.matrix_a
        vector_b = params.vector_b
        x0 = params.x0
        tol = params.tol
        niter = params.niter

        # Validar dimensiones de la matriz y vectores
        if len(matrix_a) != len(vector_b) or len(matrix_a[0]) != len(x0):
            raise HTTPException(status_code=400, detail="Dimensiones de matriz y vectores no coinciden.")

        # Ejecutar el método de Jacobi
        result = jacobi_method(matrix_a, vector_b, x0, tol, niter)

        return result

    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/incremental_search")
def get_incremental_search(params: IncrementalSearchParams):
    try:
        # Extraer los parámetros
        funcion = params.funcion
        x0 = params.x0
        intervalo = params.intervalo
        tol = params.tol
        max_iter = params.max_iter

        # Validar entrada
        if intervalo <= 0:
            raise HTTPException(status_code=400, detail="El tamaño del intervalo debe ser mayor que 0.")

        # Ejecutar el método
        resultado = busquedas_incrementales(funcion, x0, intervalo, tol, max_iter)
        return resultado

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/gaussian_elimination")
def get_gaussian_elimination(params: GaussianEliminationParams):
    try:
        # Extraer los parámetros
        A = params.A
        b = params.b

        # Validar que la matriz sea cuadrada y coincida con el vector b
        if len(A) != len(A[0]):
            raise HTTPException(status_code=400, detail="La matriz A debe ser cuadrada.")
        if len(A) != len(b):
            raise HTTPException(status_code=400, detail="El tamaño de la matriz A debe coincidir con el tamaño del vector b.")

        # Ejecutar el método de eliminación gaussiana
        resultado = eliminacion_gaussiana(A, b)
        return resultado

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/partial_pivoting")
def get_partial_pivoting(params: PartialPivotingParams):
    try:
        # Extraer los parámetros
        A = params.A
        b = params.b

        # Validar que la matriz sea cuadrada y coincida con el vector b
        if len(A) != len(A[0]):
            raise HTTPException(status_code=400, detail="La matriz A debe ser cuadrada.")
        if len(A) != len(b):
            raise HTTPException(status_code=400, detail="El tamaño de la matriz A debe coincidir con el tamaño del vector b.")

        # Ejecutar el método de pivoteo parcial
        resultado = pivoteo_parcial(A, b)
        return resultado

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/total_pivoting")
def get_total_pivoting(params: TotalPivotingParams):
    try:
        # Extraer los parámetros
        A = params.A
        b = params.b

        # Validar que la matriz sea cuadrada y coincida con el vector b
        if len(A) != len(A[0]):
            raise HTTPException(status_code=400, detail="La matriz A debe ser cuadrada.")
        if len(A) != len(b):
            raise HTTPException(status_code=400, detail="El tamaño de la matriz A debe coincidir con el tamaño del vector b.")

        # Ejecutar el método de pivoteo total
        resultado = pivoteo_total(A, b)
        return resultado

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cholesky")
def get_cholesky(params: CholeskyParams):
    try:
        # Extraer los parámetros
        A = params.A
        b = params.b

        # Validar que la matriz sea cuadrada y coincida con el vector b
        if len(A) != len(A[0]):
            raise HTTPException(status_code=400, detail="La matriz A debe ser cuadrada.")
        if len(A) != len(b):
            raise HTTPException(status_code=400, detail="El tamaño de la matriz A debe coincidir con el tamaño del vector b.")

        # Ejecutar el método de Cholesky
        resultado = cholesky_method(A, b)
        return resultado

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/gauss_seidel")
def get_gauss_seidel(params: GaussSeidelParams):
    try:
        # Extraer los parámetros
        A = params.A
        b = params.b
        x0 = params.x0
        tol = params.tol
        niter = params.niter

        # Validar que la matriz sea cuadrada y coincida con el vector b
        if len(A) != len(A[0]):
            raise HTTPException(status_code=400, detail="La matriz A debe ser cuadrada.")
        if len(A) != len(b):
            raise HTTPException(status_code=400, detail="El tamaño de la matriz A debe coincidir con el tamaño del vector b.")
        if len(A) != len(x0):
            raise HTTPException(status_code=400, detail="El tamaño de la matriz A debe coincidir con el tamaño del vector inicial x0.")

        # Ejecutar el método de Gauss-Seidel
        resultado = gauss_seidel_method(A, b, x0, tol, niter)
        return resultado

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/secant")
def get_secant(params: SecantParams):
    try:
        # Extraer los parámetros
        f = params.f
        x0 = params.x0
        x1 = params.x1
        tol = params.tol
        max_iter = params.max_iter

        # Ejecutar el método de la secante
        resultado = secant_method(f, x0, x1, tol, max_iter)
        return resultado

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
