import pandas as pd
from pybaseball import statcast
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Nos quedamos solo con columnas numéricas
datos_num = datos_mlb.select_dtypes(include=['float64', 'int64']).dropna()

# estandarizar los datos
scaler = StandardScaler()
datos_scaled = scaler.fit_transform(datos_num)

#reducción de dimensión con PCA
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(datos_scaled)

#varianza individual
varianza_individual = pca.explained_variance_ratio_
print(varianza_individual)
#varianza acumulada
varianza_acumulada = pca.explained_variance_ratio_.cumsum()
print(varianza_acumulada)


#Gráfico de varianza acumulada
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(range(1, len(varianza_acumulada)+1), varianza_acumulada, marker='o', linestyle='-')
plt.title('Varianza Acumulada por Número de Componentes')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.grid(True)
plt.show()

#decidir número de variables
import numpy as np

# Calcular el número de componentes que acumulan al menos el 70% de la varianza
n_componentes_optimo = np.argmax(varianza_acumulada >= 0.70) + 1

print(f"Se necesitan {n_componentes_optimo} componentes para explicar al menos el 70% de la varianza total.")


#Nuevo dataset reducido
from sklearn.decomposition import PCA

pca = PCA(n_components=22)
componentes = pca.fit_transform(datos_scaled)

pca_df = pd.DataFrame(componentes, columns=[f'PC{i+1}' for i in range(22)])
pca_df.head()

cargas = pd.DataFrame(
    pca.components_,
    columns=datos_num.columns,
    index=[f'PC{i+1}' for i in range(pca.n_components_)]
)

# Mostrar las 5 variables que más influyen en cada componente
for i in range(5):
    print(f"\nComponente {i+1}:")
    print(cargas.iloc[i].sort_values(ascending=False).head(5))

#Visualizar los primeros componentes
import matplotlib.pyplot as plt

plt.figure(figsize=(7,6))
plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Visualización PCA (primeros dos componentes)")
plt.show()


