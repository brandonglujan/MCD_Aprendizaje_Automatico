pip install pybaseball
pip install pyarrow
import pandas as pd
from pybaseball import statcast
from datetime import date
fecha_inicio="2025-03-27"
fecha_fin=date.today().strftime("%Y-%m-%d")
datos=statcast(start_dt=fecha_inicio,end_dt=fecha_fin)

nombre_ruta="/Users/brandon/Documents/Maestría/2T/Aprendizaje Automatizado/MLB_2025/data_mlb_2025.parquet"
datos.to_parquet(nombre_ruta,index=False)

# Cargando el csv 
import os
datos_mlb=pd.read_parquet("/Users/brandon/Documents/Maestría/2T/Aprendizaje Automatizado/MLB_2025/data_mlb_2025.parquet")

# Revisión de columnas con datos nulos
vacias=datos_mlb.isnull().all()
columnas_vacias=datos_mlb.columns[vacias]
# Eliminación de las columnas completamente vacías
datos_mlb = datos_mlb.drop(columns=columnas_vacias)
#sobreescribir el archivo para guardarlo sin las columnas eliminadas
datos_mlb.to_parquet("/Users/brandon/Documents/Maestría/2T/Aprendizaje Automatizado/MLB_2025/data_mlb_2025.parquet",index=False)
print(f"\nSe eliminaron {len(columnas_vacias)} columnas.")
print(f"Columnas restantes: {datos_mlb.shape[1]}")


