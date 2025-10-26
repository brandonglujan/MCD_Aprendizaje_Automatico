#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 22:59:08 2025

@author: brandon
"""

import numpy as np

# Crear nueva columna "High_Rating"
df_dropna["Mayor a 4"] = np.where(df["Rating"] >= 4, 1, 0)

# Revisar resultado
df[["Rating", "Mayor a 4"]].head()
