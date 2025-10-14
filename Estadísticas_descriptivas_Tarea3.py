#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 15:38:26 2025

@author: brandon
"""
df
type(df)
df.dtypes
df["Mean Salary"].describe
df.info
import statistics
statistics.mean(df["Min Salary"])
df
from scipy import stats
print(df.columns.tolist())
stat, p = stats.normaltest(df["Max Salary"])
if p > 0.05:
    print("Normal → paramétrica")
else:
    print("No normal → no paramétrica")
df_dropna=df.dropna()
df_dropna.dtypes()
df_dropna["Min Salary"].min()
df_dropna["Max Salary"].max()


from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
# 2️⃣ Seleccionar variables numéricas relevantes
varnumericas = df_dropna[["Min Salary", "Max Salary", "Mean Salary", "Rating"]]
# 4️⃣ Prueba de normalidad D'Agostino–Pearson (ideal para > 800 datos)
for col in varnumericas.columns:
    stat, p = stats.normaltest(varnumericas[col])
    print(f"{col} → estadístico={stat:.4f}, p-valor={p:.4f}")
    if p > 0.05:
        print("✅ Se considera paramétrica (distribución normal)")
    else:
        print("❌ No paramétrica (no sigue distribución normal)")
    print("-" * 50)


""" son no paramétricas """
""" transfromación a paramétricas"""

df["Log_MinSalary"] = np.log(varnumericas["Min Salary"] + 1)       # Logarítmica
df["Sqrt_MinSalary"] = np.sqrt(varnumericas["Min Salary"])         # Raíz cuadrada

df_box = df.dropna(subset=["Min Salary"]).copy()
df_box["BoxCox_MinSalary"], _ = stats.boxcox(df_box["Min Salary"] + 1) # Box-Cox
from scipy.stats import shapiro

for col in ["Min Salary", "Log_MinSalary", "Sqrt_MinSalary"]:
    data = df[col].dropna()
    stat, p = shapiro(data)
    print(f"{col}: W={stat:.4f}, p-value={p:.4f}")
 
    
moda_min_salary=df_dropna["Min Salary"].mode()
for valor in moda_min_salary:
    frecuentcia=(df_dropna["Min Salary"]==valor).sum()

frecuencia
columnas = ["Min Salary", "Max Salary", "Mean Salary"]

for col in columnas:
    modas = df_dropna[col].mode()
    print(f"\nColumna: {col}")
    print("Moda(s):", modas.tolist())
    
    for valor in modas:
        frecuencia = (df_dropna[col] == valor).sum()
        print(f"El valor {valor} aparece {frecuencia} veces")

import seaborn as sns
import matplotlib.pyplot as plt 
numericas = ["Min Salary", "Max Salary", "Mean Salary"]
categoricas = ["Industry", "Rating"]

for num_col in numericas:
    for cat_col in categoricas:
        plt.figure(figsize=(12,6))
        sns.histplot(data=df_dropna, x=num_col, hue=cat_col, multiple="stack", kde=True)
        plt.title(f"Distribución de {num_col} por {cat_col}")
        plt.xlabel(num_col)
        plt.ylabel("Cantidad")
        plt.show()
        plt.close()

import matplotlib.pyplot as plt

# Contar cuántas veces aparece cada estado
conteo = df["State"].value_counts()

# Crear gráfico de pie
plt.figure(figsize=(8,8))
plt.pie(conteo, labels=conteo.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribución de States")
plt.show()

# Selecciona solo las columnas numéricas
numericas = ["Min Salary", "Max Salary", "Mean Salary", "Rating"]

# Matriz de correlación
corr_matrix = df[numericas].corr()
print(corr_matrix)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlación")
plt.show()

"""Prueba de hipotesis""""
from scipy.stats import mannwhitneyu

# Filtrar por tamaño de empresa
California = df[df["State"] == "California"]["Mean Salary"].dropna()
Massachusetts = df[df["State"] == "Massachusetts"]["Mean Salary"].dropna()

print(f"Número de compañías pequeñas: {len(California)}")
print(f"Número de compañías grandes: {len(Massachusetts)}")

stat, p = mannwhitneyu(California, Massachusetts, alternative="two-sided")
print(f"Mann-Whitney U = {stat:.4f}, p-value = {p:.4f}")

if p <= 0.05:
    print("Diferencia significativa en el sueldo promedio entre California y Massachusetts")
else:
    print("No hay diferencia significativa en el sueldo promedio entre California y Massachusetts")



