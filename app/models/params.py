from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

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