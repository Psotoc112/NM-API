from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from typing import List

class biseccionParams(BaseModel):
    eqn: str
    xi: float
    xf: float
    tol: float

class raicesMParams(BaseModel):
    eqn: str
    eqn1: str
    eqn2: str
    xo: float
    tol: float

class reglaFalsaParams(BaseModel):
    eqn: str
    xi: float
    xf: float
    tol: float

class newtonParams(BaseModel):
    eqn: str
    eqn1: str
    xo: float
    tol: float

class puntoFParams(BaseModel):
    eqn: str
    eqn2: str
    valorA: float
    tol: float

class spline1Params(BaseModel):
    x: List[float]
    y: List[float]

class croutParams(BaseModel):
    A: List[List[float]]
    b: List[float]

class dolittleParams(BaseModel):
    A: List[List[float]]
    b: List[float]