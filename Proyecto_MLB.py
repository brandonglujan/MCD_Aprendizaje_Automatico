##### objetivo predecir si el bateador har谩 swing o no, considerando el tipo de pitcheo.
pip install pybaseball
pip install pyarrow
import pandas as pd
from pybaseball import statcast
from datetime import date
fecha_inicio="2024-03-20"
fecha_fin="2024-10-30"
datos=statcast(start_dt=fecha_inicio,end_dt=fecha_fin)
nombre_ruta="/Users/brandon/Documents/Maestri虂a/2T/Aprendizaje Automatizado/MLB_2025/data_mlb_2024.parquet"
datos.to_parquet(nombre_ruta,index=False)

# Cargando el parquet 
import os
datos_mlb=pd.read_parquet("/Users/brandon/Documents/Maestri虂a/2T/Aprendizaje Automatizado/MLB_2025/data_mlb_2024.parquet")

# Revisi贸n de columnas con datos nulos
vacias=datos_mlb.isnull().all()
columnas_vacias=datos_mlb.columns[vacias]
# Eliminaci贸n de las columnas completamente vac铆as
datos_mlb = datos_mlb.drop(columns=columnas_vacias)
#sobreescribir el archivo para guardarlo sin las columnas eliminadas
datos_mlb.to_parquet("/Users/brandon/Documents/Maestri虂a/2T/Aprendizaje Automatizado/MLB_2025/data_mlb_2024.parquet",index=False)
print(f"\nSe eliminaron {len(columnas_vacias)} columnas.")
print(f"Columnas restantes: {datos_mlb.shape[1]}")


# Crear variable binaria 'swing' a partir de la descripci贸n del lanzamiento
# Casos donde el bateador hace swing
swing_terms = [
    'swinging_strike',       # strike con swing
    'foul',                  # foul siempre implica swing
    'hit_into_play',         # pelota puesta en juego
    'foul_tip',              # foul siempre implica swing
    'foul_bunt',             # foul siempre implica swing
    'swinging_strike_blocked' # swing fallado bloqueado
]

# Crear variable swing = 1 si el evento pertenece a la lista
datos_mlb['swing'] = datos_mlb['description'].isin(swing_terms).astype(int)

# Verificar distribuci贸n
print(datos_mlb['swing'].value_counts(normalize=True))

# Revisar tipos de pitcheo 煤nicos
print(datos_mlb['pitch_type'].value_counts(dropna=False))

# Eliminar filas sin tipo de pitcheo
datos_mlb = datos_mlb.dropna(subset=['pitch_type'])

# Guardar dataset limpio y con variable objetivo
nombre_final = "/Users/brandon/Documents/Maestr铆a/2T/Aprendizaje Automatizado/MLB_2025/data_mlb_swing.parquet"
datos_mlb.to_parquet(nombre_final, index=False)

##ABRIR DATOS
ruta = "/Users/brandon/Documents/Maestr韆/2T/Aprendizaje Automatizado/MLB_2025/data_mlb_swing.parquet"
datos_mlb = pd.read_parquet(ruta)

ruta = "/Users/brandon/Documents/Maestr韆/2T/Aprendizaje Automatizado/MLB_2025/data_mlb_swing.parquet"
datos_mlb = pd.read_parquet(ruta)

####################################
############ ALGORITMOS ############
####################################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Codificaci贸n one-hot de la variable 'pitch_type'
encoder = OneHotEncoder(drop='first')
X = encoder.fit_transform(datos_mlb[['pitch_type']])
y = datos_mlb['swing']

# Divisi贸n entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
####################################
# MODELO 1: REGRESI脫N LOG脥STICA
####################################
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\n===== REGRESI脫N LOG脥STICA =====")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

####################################
# MODELO 2: 脕RBOL DE DECISI脫N
####################################
tree_model = DecisionTreeClassifier(
    criterion='gini', 
    max_depth=None, 
    random_state=42
)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

print("\n===== 脕RBOL DE DECISI脫N =====")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))
print("\nMatriz de confusi贸n:\n", confusion_matrix(y_test, y_pred_tree))

####################################
# VISUALIZACI脫N DEL 脕RBOL
####################################
plt.figure(figsize=(18,10))
plot_tree(
    tree_model,
    feature_names=encoder.get_feature_names_out(['pitch_type']),
    class_names=['No Swing', 'Swing'],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("脕rbol de Decisi贸n: Predicci贸n del Swing seg煤n el Tipo de Pitcheo")
plt.show()

####################################
# IMPORTANCIA DE VARIABLES
####################################
importances = tree_model.feature_importances_
feature_names = encoder.get_feature_names_out(['pitch_type'])
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.bar(range(len(importances)), importances[sorted_idx])
plt.xticks(range(len(importances)), feature_names[sorted_idx], rotation=45)
plt.title("Importancia de cada tipo de pitcheo en la predicci髇 del swing")
plt.ylabel("Importancia")
plt.show()

####################################
# GUARDAR RESULTADOS RESUMIDOS
####################################
resultados = pd.DataFrame({
    'Modelo': ['Regresi髇 Log韘tica', '羠bol de Decisi髇'],
    'Accuracy': [accuracy_score(y_test, y_pred_log), accuracy_score(y_test, y_pred_tree)]
})
print("\nResumen de desempe駉 de modelos:")
print(resultados)

####################################
# MATRIZ DE CONFUSI脫N HEAT 
####################################
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred: No Swing', 'Pred: Swing'],
            yticklabels=['Real: No Swing', 'Real: Swing'])
plt.title("Matriz de confusi髇 - Regresi髇 log韘tica")
plt.show()

cm2 = confusion_matrix(y_test, y_pred_tree)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred: No Swing', 'Pred: Swing'],
            yticklabels=['Real: No Swing', 'Real: Swing'])
plt.title("Matriz de confusi髇 - 羠bol de decisi髇")
plt.show()


