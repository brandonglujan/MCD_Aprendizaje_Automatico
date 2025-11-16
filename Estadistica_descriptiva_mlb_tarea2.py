# Estadística descriptiva
# No se revisa normalidad debido a que se tiene muestra grande y el total de mi población es la temporada completa
import numpy as np
import pandas as pd
import statistics
promediovel=datos_mlb["release_speed"].mean
velefec=np.nanmean(datos_mlb["effective_speed"])

print(promediovel)
print(velefec)

numericas=datos_mlb.select_dtypes(include=['number'])
correlacion=numericas.corr()

from pybaseball import playerid_lookup
from pybaseball import statcast_pitcher

# Encontrar playerID de Shohei Ohtani
ohtani_id=playerid_lookup('ohtani', 'shohei')

# Su MLBAM ID is 660271, así que calculamos la media de cada uno de los lanzamientos de Ohtani 
ohtani_stats = statcast_pitcher(fecha_inicio, fecha_fin, 660271)
ohtani_stats.groupby("pitch_type").release_speed.agg("mean")

# GRAFICANDO DATOS DE OHTANI 

from pybaseball import statcast_pitcher
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Shohei Ohtani - temporada 2023
ohtani_stats = statcast_pitcher(fecha_inicio, "2025-09-30", 660271)
ohtani_stats.head()


### GRAFICO DE VELOCIDAD PROMEDIO ###
plt.figure(figsize=(8, 4))
sns.barplot(
    data=ohtani_stats,
    x="pitch_type",
    y="release_speed",
    estimator="mean",
    ci=None,
    palette="viridis"
)
plt.title("Velocidad promedio por tipo de lanzamiento")
plt.xlabel("Tipo de lanzamiento")
plt.ylabel("Velocidad promedio (mph)")
plt.show()
######################################

####### MAPA DE CALOR #######
plt.figure(figsize=(6, 6))
sns.kdeplot(
    data=ohtani_stats,
    x="plate_x",
    y="plate_z",
    fill=True,
    cmap="Reds",
    thresh=0.05
)
plt.title("Mapa de calor de ubicación de lanzamientos")
plt.xlabel("Posición horizontal (pies)")
plt.ylabel("Posición vertical (pies)")
plt.axhline(1.5, color="black", linestyle="--")
plt.axhline(3.5, color="black", linestyle="--")
plt.axvline(-0.83, color="black", linestyle="--")
plt.axvline(0.83, color="black", linestyle="--")
plt.show()
######################################

plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=ohtani_stats,
    x="plate_x",  # Posición horizontal
    y="plate_z",  # Posición vertical
    hue="pitch_type",  # Color por tipo de lanzamiento
    alpha=0.6
)
plt.title("Ubicación de lanzamientos de Shohei Ohtani (Statcast 2025)")
plt.xlabel("Posición horizontal (pies)")
plt.ylabel("Posición vertical (pies)")

# Cuadro del strike zone (zona de strike) 
# Se utilizan estas medidas de forma estándar, si quisiera un bateador en específico se usa sz_top y sz_bot del bateador en específico
plt.axhline(1.5, color="black", linestyle="--")
plt.axhline(3.5, color="black", linestyle="--")
plt.axvline(-0.83, color="black", linestyle="--")
plt.axvline(0.83, color="black", linestyle="--")

plt.legend(title="Tipo de lanzamiento")
plt.show()







