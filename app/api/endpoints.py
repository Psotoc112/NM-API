# app/api/endpoints.py

from fastapi import APIRouter, HTTPException
from app.services.methods import *
from app.models.params import *

router = APIRouter()

@router.get("/biseccion/{eqn}/{xi}/{xf}/{tol}")
def get_biseccion(eqn: str, xi: float, xf: float, tol: float):
    result = biseccion(eqn,xi,xf,tol)
    return {"result": result}

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
def get_reglaFalsa(params: biseccionParams):

    eqn = params.eqn
    xi = params.xi
    xf = params.xf
    tol = params.tol
    result = reglaFalsa(eqn,xi,xf,tol)
    return {"result": result}

#@router.get("/add/{a}/{b}")
#def get_add(a: int, b: int):
#    result = add(a,b)
#    return {"result": result} 