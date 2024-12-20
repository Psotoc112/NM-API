from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from typing import List

class biseccionParams(BaseModel):
    eqn: str
    xi: float
    xf: float
    tol: float
    niter: int

class raicesMParams(BaseModel):
    eqn: str
    eqn1: str
    eqn2: str
    xo: float
    tol: float
    niter: int

class reglaFalsaParams(BaseModel):
    eqn: str
    xi: float
    xf: float
    tol: float
    niter: int

class newtonParams(BaseModel):
    eqn: str
    eqn1: str
    xo: float
    tol: float
    niter: int

class puntoFParams(BaseModel):
    eqn: str
    eqn2: str
    valorA: float
    tol: float
    niter: int

class spline1Params(BaseModel):
    x: List[float]
    y: List[float]

class vanderParams(BaseModel):
    x: List[float]
    y: List[float]

class croutParams(BaseModel):
    A: List[List[float]]
    b: List[float]

class dolittleParams(BaseModel):
    A: List[List[float]]
    b: List[float]

class LUParams(BaseModel):
    A: List[List[float]]
    b: List[float]

class LUParams(BaseModel):
    A: List[List[float]]
    b: List[float]

class NewtonParams(BaseModel):
    x_valor: list[float]
    y_valor: list[float]

class LagrangeParams(BaseModel):
    x: List[float]
    y: List[float]

class JacobiParams(BaseModel):
    matrix_a: List[List[float]]  
    vector_b: List[float]        
    x0: List[float]              
    tol: float                  
    niter: int                   

class IncrementalSearchParams(BaseModel):
    funcion: str     
    x0: float        
    intervalo: float 
    tol: float       
    max_iter: int     

class GaussianEliminationParams(BaseModel):
    A: List[List[float]]  
    b: List[float]     

class PartialPivotingParams(BaseModel):
    A: List[List[float]]  
    b: List[float] 

class TotalPivotingParams(BaseModel):
    A: List[List[float]]  
    b: List[float] 

class CholeskyParams(BaseModel):
    A: List[List[float]] 
    b: List[float]

class GaussSeidelParams(BaseModel):
    A: List[List[float]]  
    b: List[float]       
    x0: List[float]      
    tol: float          
    niter: int

class SecantParams(BaseModel):
    f: str        
    x0: float     
    x1: float    
    tol: float    
    max_iter: int
