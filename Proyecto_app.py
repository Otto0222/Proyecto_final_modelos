import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Scikit-learn imports
from sklearn import tree
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.inspection import PartialDependenceDisplay
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Imbalanced learn
from imblearn.over_sampling import RandomOverSampler


# Descargar el dataset de Kaggle (pima-indians-diabetes-database)
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")


# Cargar el archivo CSV desde el path
diabetes = pd.read_csv(f"{path}/diabetes.csv", na_values=['?'])


# Ver la distribución de las clases en la columna 'Outcome' de la base de datos original
class_distribution = diabetes['Outcome'].value_counts()


# Filtrar las clases 0 y 1
class_0 = diabetes[diabetes['Outcome'] == 0]
class_1 = diabetes[diabetes['Outcome'] == 1]

# Eliminar aleatoriamente 200 ejemplos de la clase 0 para balancear
class_0_reduced = class_0.sample(n=len(class_0) - 200, random_state=42)

# Combinar las clases balanceadas
diabetes_balanced = pd.concat([class_0_reduced, class_1])


# Balanceo de datos
# Manejo de valores nulos (se eliminan filas con NaN)
diabetes_balanced.dropna(inplace=True)

# Separar variables predictoras (X) y variable objetivo (y)
X = diabetes_balanced.drop(columns=['Outcome'])  # 'Outcome' es la variable objetivo
y = diabetes_balanced['Outcome']


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

# Convertir resultados en un DataFrame
results_df = pd.DataFrame(results)


# Evaluar el mejor modelo en el conjunto de prueba
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)  # Sensibilidad (recall)



# Graficar el árbol de decisión en caso de que este sea el mejor modelo
if isinstance(best_model, Pipeline):
    # Acceder al árbol de decisión en el pipeline
    tree_model = best_model.named_steps['model']  # 'model' is the step name from the pipeline


    
# Mostrar la matriz de confusión del modelo con mejor resultado
# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Obtener el nombre del modelo de best_model
model_name = best_model.named_steps["model"].__class__.__name__  # Obtener el nombre del modelo


# Crear la aplicación de Streamlit
# Title of the Streamlit app
st.title("Diabetes Prediction Model Results")

#Create tabs
tab1, tab2, tab3 = st.tabs(["EDA", "Classification", "Clustering"])


#-------------------------------- Tab 1: EDA --------------------------------
with tab1:
    st.subheader("📊 Model Results")
    st.write("Show model performance, accuracy, recall, confusion matrix, etc.")


#-------------------------------- Tab 2: Classification --------------------------------
with tab2:

    diabetes['Outcome'] = diabetes['Outcome'].astype(int)

    # Show dataset preview
    if st.checkbox("Show raw data"):
        st.write(diabetes.head())

    # Class distribution
    st.subheader("Class Distribution in Original Dataset")
    class_distribution = diabetes['Outcome'].value_counts()
    st.bar_chart(class_distribution)

    # Balancing dataset with undersampling
    class_0 = diabetes[diabetes['Outcome'] == 0]
    class_1 = diabetes[diabetes['Outcome'] == 1]
    class_0_reduced = class_0.sample(n=len(class_0) - 200, random_state=42)
    diabetes_balanced = pd.concat([class_0_reduced, class_1])

    # Show balanced class distribution
    st.subheader("Class Distribution After Balancing")
    balanced_class_distribution = diabetes_balanced['Outcome'].value_counts()
    st.bar_chart(balanced_class_distribution)


    st.subheader("Model Performance")
    st.write(results_df)


    #Decision tree
    st.subheader("Decision tree result")
    # Acceder al árbol de decisión en el pipeline
    tree_model = best_model.named_steps['model']  # 'model' is the step name from the pipeline

    # Graficar el árbol de decisión
    fig, ax = plt.subplots(figsize=(12, 8))  # Define figure and axis
    plot_tree(tree_model, filled=True, feature_names=X.columns, rounded=True, ax=ax)  # Use ax
    st.pyplot(fig)  # Display in Streamlit

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


#-------------------------------- Tab 1: EDA --------------------------------
with tab3:
    
    dataset = diabetes
    
    # Class distribution
    st.subheader("Class Distribution in Original Dataset")
    class_distribution = diabetes['Outcome'].value_counts()
    st.bar_chart(class_distribution)

    #Balancear los datos con oversampling
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(dataset.drop(columns=['Outcome']), dataset['Outcome'])

    # Crear un nuevo DataFrame equilibrado
    df = pd.DataFrame(X_resampled, columns=dataset.drop(columns=['Outcome']).columns)
    df['Outcome'] = y_resampled

    #Shows the pairplot of the variables 
    st.subheader("variables - pairplot")
    fig = sns.pairplot(df, hue="Outcome")
    st.pyplot(fig)

    #Eliminar variables con poco peso
    df = df.drop(columns=['SkinThickness', 'Insulin', 'DiabetesPedigreeFunction'])  # Eliminar variables del dataframe


    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df.drop(columns=['Outcome'], errors='ignore'))


    # Reducción de dimensionalidad con PCA para visualización
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)


    # Definir min_samples (prueba con 5 o 10)
    min_samples = 10

    # Calcular las distancias al k-ésimo vecino más cercano
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(data_scaled)
    distances, indices = neighbors_fit.kneighbors(data_scaled)

    # Ordenar y graficar las distancias
    st.subheader("Gráfico para encontrar Epsilon (k-Distance)")
    fig, ax = plt.subplots()
    ax.plot(distances)
    ax.set_xlabel("Sorted Points")
    ax.set_ylabel(f"Distance to {min_samples}-th Nearest Neighbor")
    ax.set_title("k-Distance Graph for Finding Epsilon")

    # Show the plot in Streamlit
    st.pyplot(fig)
