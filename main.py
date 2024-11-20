from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router

app = FastAPI()
app.include_router(api_router)
# Configuración de CORS
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:3000",
    "https://nm-api.onrender.com",
    "matrixengine.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)
