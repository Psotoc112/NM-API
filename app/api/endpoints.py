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
    result = newton(eqn,eqn2,a,tol)
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