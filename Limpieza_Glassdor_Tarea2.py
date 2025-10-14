import pandas as pd
import numpy as np
df=pd.read_csv('/Users/brandon/Documents/Maestría/2T/Aprendizaje Automatizado/Data science salaries glassdoor/glassdoor_jobs.csv',encoding='unicode_escape')
df.info()

"""Creación de columnas min, max y media con la columna salary estimate"""
df["Salary clean"] = df["Salary Estimate"].str.replace(r"[^\d-]", "", regex=True)
df[["Min Salary","Max Salary"]] = df["Salary clean"].str.split('-',expand=True)

"""Se convierte a númerico"""
df["Min Salary"] = pd.to_numeric(df["Min Salary"]) 
df["Max Salary"] = pd.to_numeric(df["Max Salary"])

"""Axis=1 calcula la media por fila y no por columnnas"""
df["Mean Salary"] = df[["Min Salary", "Max Salary"]].mean(axis=1) 

"""Le quito la calificación de la compañía para que no haga ruido"""
df["Company Name"]=df["Company Name"].str.split("\n").str[0]

df.info()
df["Mean Salary"] =df["Media"] 

df.drop("Media",axis=1,inplace=True)

print(df.head)

df[["City","State"]]=df["Location"].str.split(", ",expand=True,n=1)

state_names = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming"
}

df["State"]=df["State"].map(state_names)
df.columns.values[0]="Id"
