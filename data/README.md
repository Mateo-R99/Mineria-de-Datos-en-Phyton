# Datos

**Archivo:** `titanic.csv`

Dataset histórico del Titanic usado en las prácticas anteriores del equipo (calidad de datos y minería predictiva). Contiene 891 registros de pasajeros con 12 columnas:

| Columna | Descripción |
|---|---|
| `PassengerId` | Identificador único del pasajero (no se usa como predictor) |
| `Survived` | Variable objetivo: 1 = sobrevivió, 0 = no sobrevivió |
| `Pclass` | Clase del boleto (1 = primera, 2 = segunda, 3 = tercera) |
| `Name` | Nombre del pasajero (no se usa como predictor) |
| `Sex` | Sexo (`male` / `female`) |
| `Age` | Edad en años (177 valores faltantes, imputados con la mediana en el pipeline) |
| `SibSp` | Número de hermanos/as o cónyuges a bordo |
| `Parch` | Número de padres/hijos a bordo |
| `Ticket` | Número de boleto (no se usa como predictor) |
| `Fare` | Tarifa pagada |
| `Cabin` | Número de camarote (77% de valores faltantes; se descarta del análisis) |
| `Embarked` | Puerto de embarque (`S` = Southampton, `C` = Cherbourg, `Q` = Queenstown; 2 valores faltantes, imputados con la moda) |

No se modificó ni se inventó ningún dato: el archivo es exactamente el que el equipo viene usando desde las prácticas anteriores de calidad de datos y minería predictiva.
