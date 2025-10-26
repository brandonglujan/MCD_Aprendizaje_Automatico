# Aprendizaje Automatico
## Eric Brandon García Luján
### Tarea 4

Para fines de esta materia, se ha optado por elegir una librería llamada pybaseball como base de datos, la cual contiene la información actualizada de las grandes ligas de beísbol con el fin de poder explorar algunas de las tantas variables que se tienen dentro de un juego. En este deporte existe algo llamado _sabermetría_ (SABRmetrics) que básicamente es el análisis estadístico del beísbol, el cual al día de hoy es lo más usado para aprender sobre las estadísticas de los juegos de beísbol.

En este escrito se abordará el [artículo](Articulo_SeleccionarVariables_Tarea4.pdf) escrito por Shu-Fen Li, Mei-Ling Huang y Yun-Zhi Li que lleva por título _"Use of Machine Learning and Deep Learning to Predict the Outcomes of Major League Baseball Matches"_. 
En este artículo, también recopilan información de la misma sobre la cual estaremos trabajando (_PyBaseball_).

En el documento se menciona que el método para seleccionar caraterísticas usado usado en esta investigación es ReliefF que es un método de filtro.
ReliefF tiene como proposito identificar qué variables son más relevantes para predecir la variable objetivo, como en alguna parte de nuestro código donde queremos visualizar si fue strike o no. Este algoritmo se usa en problemas de multiclasificación y regresión ya que obtiene repetidamente muestras del conjunto de entrenamiento y calcula la diferencia entre las instancias para encontrar los vecinos más cercanos (las filas con los datos más parecidos entre sí).

ReliefF:
 - No mide correlación lineal
 - Mide la capacidad de una variable para distinguir entre clases, incluso si la relación es no lineal.
 - Útil en datasets complejos o con mucho ruido. 

 Para este artículo los autores utilizan Weka que es una interfaz más amigable para trabajar con datos.

 Nuestro dataset es un dataset de una libreria completa sobre el total de acciones dentro de una liga completa por lo que determinar las características más relevantes no es posible en el punto en que nos encontramos actualmente en el tetramestra ya que dependerá del tipo de análisis que se desee realizar; si analizamos a pitchers por ejemplo, las variables más importantes tendrán que ser realacionadas al tipo de pitcheo o velocidad, pero si analizamos alguna jugada tal vez estas variables no sean tan relevantes.


Fuente:
Huang, M.-L.; Li, Y.-Z. Use
of Machine Learning and Deep
Learning to Predict the Outcomes of
Major League Baseball Matches. Appl.
Sci. 2021, 11, 4499. https://doi.org/
10.3390/app11104499