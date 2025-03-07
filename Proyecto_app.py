import kagglehub
import pandas as pd
from sklearn import tree
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.inspection import PartialDependenceDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Descargar el dataset de Kaggle (pima-indians-diabetes-database)
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")

# Cargar el archivo CSV desde el path
diabetes = pd.read_csv(f"{path}/diabetes.csv", na_values=['?'])


# Ver la distribución de las clases en la columna 'Outcome' de la base de datos original
class_distribution = diabetes['Outcome'].value_counts()

# Graficar la distribución de clases
plt.figure(figsize=(6, 4))
sns.barplot(x=class_distribution.index, y=class_distribution.values, palette="Blues_d", hue = class_distribution.index)
plt.title("Distribución de Clases en el Dataset Original")
plt.xlabel("Clase")
plt.ylabel("Número de sujetos")
plt.xticks(ticks=[0, 1], labels=["Sin Diabetes (0)", "Con Diabetes (1)"])
plt.show()


# Mostrar la distribución original de las clases
print("Distribución original de las clases:")
print(diabetes['Outcome'].value_counts())

# Filtrar las clases 0 y 1
class_0 = diabetes[diabetes['Outcome'] == 0]
class_1 = diabetes[diabetes['Outcome'] == 1]

# Eliminar aleatoriamente 200 ejemplos de la clase 0
class_0_reduced = class_0.sample(n=len(class_0) - 232, random_state=42)

# Combinar las clases balanceadas
diabetes_balanced = pd.concat([class_0_reduced, class_1])

# Ver la distribución de clases después de eliminar ejemplos de la clase 0
print("\nDistribución de las clases después de balancear:")
print(diabetes_balanced['Outcome'].value_counts())


# Manejo de valores nulos (se eliminan filas con NaN)
diabetes.dropna(inplace=True)

# Separar variables predictoras (X) y variable objetivo (y)
X = diabetes.drop(columns=['Outcome'])  # 'Outcome' es la variable objetivo
y = diabetes['Outcome']

# Balanceo de datos
# Manejo de valores nulos (se eliminan filas con NaN)
diabetes_balanced.dropna(inplace=True)

# Separar variables predictoras (X) y variable objetivo (y)
X = diabetes_balanced.drop(columns=['Outcome'])  # 'Outcome' es la variable objetivo
y = diabetes_balanced['Outcome']

# División con StratifiedShuffleSplit para balancear clases
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=22)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Definir modelos y sus hiperparámetros
models = {
    "LogisticRegression": {
        "model": LogisticRegression(),
        "params": {
            "model__C": [0.01, 0.1, 1, 10, 100],  # Añadimos más valores para C
            "model__solver": ["liblinear", "saga"],  # Opción 'saga' para problemas grandes
            "model__penalty": ["l1", "l2"],  # Probar diferentes penalizaciones
            "model__max_iter": [100, 200]  # Iteraciones adicionales si es necesario
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {
            "model__n_estimators": [50, 100, 200],  # Más árboles
            "model__max_depth": [3, 5, 10, None],  # Profundidades adicionales
            "model__min_samples_split": [2, 5, 10],  # Divisiones mínimas para cada nodo
            "model__min_samples_leaf": [1, 2, 5],  # Hojas mínimas para cada nodo
            "model__bootstrap": [True, False]  # Si usar bootstrap o no
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "model__max_depth": [3, 5, 10, None],  # Más profundidades
            "model__min_samples_split": [2, 5, 10],  # Divisiones mínimas
            "model__min_samples_leaf": [1, 2, 5],  # Hojas mínimas
            "model__criterion": ["gini", "entropy"]  # Diferentes criterios de división
        }
    }
}

# Ejecutar validación cruzada para cada modelo, eligiendo según Recall
best_model = None
best_score = 0
results = []

for name, config in models.items():
    print(f"\n🔍 Evaluando {name} con validación cruzada...")

    # Crear pipeline con estandarización + modelo
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", config["model"])])

    # GridSearch con validación cruzada (CV=10)
    grid_search = GridSearchCV(pipeline, config["params"], cv=10, scoring="recall", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Guardar los resultados en un DataFrame
    results.append({
        "Modelo": name,
        "Mejor Recall CV": grid_search.best_score_,
        "Mejores Hiperparámetros": grid_search.best_params_
    })

    # Guardar el mejor modelo según recall
    if grid_search.best_score_ > best_score:
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_

    print(f"✅ {name} - Mejor Recall en CV: {grid_search.best_score_:.4f}")

# Convertir resultados en un DataFrame y mostrar
results_df = pd.DataFrame(results)
print("\nResultados de los Modelos y Mejores Hiperparámetros:")
print(results_df)

# Evaluar el mejor modelo en el conjunto de prueba
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)  # Sensibilidad (recall)

# Mostrar resultados finales
print("\n🎯 Mejor modelo final:", best_model)
print(f"🏆 Mejor Recall en validación: {best_score:.4f}")
print(f"📊 Accuracy en test: {test_accuracy:.4f}")
print(f"🔍 Sensibilidad (Recall) en test: {test_recall:.4f}")

# Graficar el árbol de decisión en caso de que este sea el mejor modelo
if isinstance(best_model, Pipeline) and isinstance(best_model.named_steps['model'], DecisionTreeClassifier):
    # Access the decision tree classifier inside the pipeline
    tree_model = best_model.named_steps['model']  # 'model' is the step name from the pipeline

    # Plot the decision tree
    plt.figure(figsize=(12, 8))  # Adjust the size of the plot
    plot_tree(tree_model, filled=True, feature_names=X.columns, rounded=True)
    plt.show()
    
    
# Mostrar la matriz de confusión del modelo con mejor resultado
# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Obtener el nombre del modelo de best_model
model_name = best_model.named_steps["model"].__class__.__name__  # Obtener el nombre del modelo

# Mostrar la matriz de confusión con un heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.title(f"Matriz de Confusión de {model_name}")  # Corregir el uso del f-string
plt.xlabel("Predicción")
plt.ylabel("Verdadero")
plt.show()


# División con StratifiedKFold (K-folds)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=22)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Definir modelos y sus hiperparámetros
models = {
    "LogisticRegression": {
        "model": LogisticRegression(),
        "params": {
            "model__C": [0.01, 0.1, 1, 10, 100],  # Añadimos más valores para C
            "model__solver": ["liblinear", "saga"],  # Opción 'saga' para problemas grandes
            "model__penalty": ["l1", "l2"],  # Probar diferentes penalizaciones
            "model__max_iter": [100, 200]  # Iteraciones adicionales si es necesario
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {
            "model__n_estimators": [50, 100, 200],  # Más árboles
            "model__max_depth": [3, 5, 10, None],  # Profundidades adicionales
            "model__min_samples_split": [2, 5, 10],  # Divisiones mínimas para cada nodo
            "model__min_samples_leaf": [1, 2, 5],  # Hojas mínimas para cada nodo
            "model__bootstrap": [True, False]  # Si usar bootstrap o no
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "model__max_depth": [3, 5, 10, None],  # Más profundidades
            "model__min_samples_split": [2, 5, 10],  # Divisiones mínimas
            "model__min_samples_leaf": [1, 2, 5],  # Hojas mínimas
            "model__criterion": ["gini", "entropy"]  # Diferentes criterios de división
        }
    }
}

# Ejecutar validación cruzada para cada modelo, eligiendo según Recall
best_model = None
best_score = 0
results = []

for name, config in models.items():
    print(f"\n🔍 Evaluando {name} con validación cruzada...")

    # Crear pipeline con estandarización + modelo
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", config["model"])])

    # GridSearch con validación cruzada (CV=10)
    grid_search = GridSearchCV(pipeline, config["params"], cv=10, scoring="recall", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Guardar los resultados en un DataFrame
    results.append({
        "Modelo": name,
        "Mejor Recall CV": grid_search.best_score_,
        "Mejores Hiperparámetros": grid_search.best_params_
    })

    # Guardar el mejor modelo según recall
    if grid_search.best_score_ > best_score:
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_

    print(f"✅ {name} - Mejor Recall en CV: {grid_search.best_score_:.4f}")

# Convertir resultados en un DataFrame y mostrar
results_df = pd.DataFrame(results)
print("\nResultados de los Modelos y Mejores Hiperparámetros:")
print(results_df)

# Evaluar el mejor modelo en el conjunto de prueba
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)  # Sensibilidad (recall)

# Mostrar resultados finales
print("\n🎯 Mejor modelo final:", best_model)
print(f"🏆 Mejor Recall en validación: {best_score:.4f}")
print(f"📊 Accuracy en test: {test_accuracy:.4f}")
print(f"🔍 Sensibilidad (Recall) en test: {test_recall:.4f}")

# Graficar el árbol de decisión en caso de que este sea el mejor modelo
if isinstance(best_model, Pipeline):
    # Acceder al árbol de decisión en el pipeline
    tree_model = best_model.named_steps['model']  # 'model' is the step name from the pipeline

    # Graficar el árbol de decisión
    plt.figure(figsize=(12, 8))  # Adjust the size of the plot
    plot_tree(tree_model, filled=True, feature_names=X.columns, rounded=True)
    plt.show()
    
    
# Mostrar la matriz de confusión del modelo con mejor resultado
# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Obtener el nombre del modelo de best_model
model_name = best_model.named_steps["model"].__class__.__name__  # Obtener el nombre del modelo

# Mostrar la matriz de confusión con un heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.title(f"Matriz de Confusión de {model_name}")  # Corregir el uso del f-string
plt.xlabel("Predicción")
plt.ylabel("Verdadero")
plt.show()


# Title of the Streamlit app
st.title("Diabetes Prediction Model Results")

# Load dataset
# Title of the Streamlit app
st.title("Diabetes Prediction Model Results")

diabetes['Outcome'] = diabetes['Outcome'].astype(int)

# Show dataset preview
if st.checkbox("Show raw data"):
    st.write(diabetes.head())

# Class distribution
st.subheader("Class Distribution in Original Dataset")
class_distribution = diabetes['Outcome'].value_counts()
st.bar_chart(class_distribution)

# Balancing dataset
class_0 = diabetes[diabetes['Outcome'] == 0]
class_1 = diabetes[diabetes['Outcome'] == 1]
class_0_reduced = class_0.sample(n=len(class_0) - 232, random_state=42)
diabetes_balanced = pd.concat([class_0_reduced, class_1])

# Show balanced class distribution
st.subheader("Class Distribution After Balancing")
balanced_class_distribution = diabetes_balanced['Outcome'].value_counts()
st.bar_chart(balanced_class_distribution)


results_df = best_model
st.subheader("Model Performance")
st.write(results_df)

# Confusion matrix
st.subheader("Confusion Matrix of Best Model")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Show accuracy and recall
st.subheader("Final Model Performance")
st.write(f"**Best Model:** {model_name}")
st.write(f"**Accuracy:** {test_accuracy:.4f}")
st.write(f"**Recall:** {test_recall:.4f}")
