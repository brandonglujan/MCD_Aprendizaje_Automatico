
#  Selección de variables relevantes con Mutual Information

import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import LabelEncoder


# --- Se crea la variable objetivo (strike = 1, no strike = 0) ---
ohtani_stats["es strike"] = ohtani_stats["description"].isin([
    "called_strike", "swinging_strike", "foul_tip", "swinging_strike_blocked"
]).astype(int)

# --- Se selecciona variables numéricas que podrían influir ---
X = ohtani_stats[[
    "release_speed", "release_spin_rate", "plate_x", "plate_z",
    "pfx_x", "pfx_z", "vx0", "vy0", "vz0"
]].copy()

y = ohtani_stats["es strike"]

# --- Se eliminan filas con valores faltantes ---
X = X.fillna(0)
y = y.fillna(0)

# --- Calcular la información mutua ---
selector = SelectKBest(score_func=mutual_info_classif, k='all')
selector.fit(X, y)

# --- Se crea DataFrame con los puntajes ---
importances = pd.DataFrame({
    'Variable': X.columns,
    'Importancia (Mutual Info)': selector.scores_
}).sort_values(by='Importancia (Mutual Info)', ascending=False)

print("\n📈 Importancia de cada variable respecto al strike:\n")
print(importances)

# --- Resultados ---
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.barh(importances["Variable"], importances["Importancia (Mutual Info)"], color="crimson")
plt.xlabel("Importancia (Mutual Information)")
plt.title("Relación entre variables del lanzamiento y strikes")
plt.gca().invert_yaxis()
plt.show()


