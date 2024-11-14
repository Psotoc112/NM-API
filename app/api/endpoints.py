# app/api/endpoints.py

from fastapi import APIRouter, HTTPException
from app.services.methods import *

router = APIRouter()

@router.get("/biseccion/{eqn}/{xi}/{xf}/{tol}")
def get_biseccion(eqn: str, xi: float, xf: float, tol: float):
    result = biseccion(eqn,xi,xf,tol)
    return {"result": result}

#@router.get("/add/{a}/{b}")
#def get_add(a: int, b: int):
#    result = add(a,b)
#    return {"result": result} 