import kagglehub
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

# Scikit-learn imports
from sklearn import tree
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, silhouette_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN

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
    if st.checkbox("Mostrar dataframe"):
        st.write(diabetes.head())

    # Class distribution
    st.subheader("Distribución de clases en Dataset original")
    class_distribution = diabetes['Outcome'].value_counts()
    st.bar_chart(class_distribution)

    st.markdown("## Balanceo de Datos")

    st.markdown("""
    Al observar la distribución de las clases en el dataset, se puede notar que los datos están desbalanceados, habiendo casi el doble de sujetos **sin diabetes** que **con diabetes**.  

    Debido a esto, se utiliza la técnica de **undersampling**, seleccionando una muestra aleatoria de los sujetos **sin diabetes** y tomando a todos los sujetos **con diabetes** para construir un nuevo dataset, de forma que ambas clases tengan el mismo número de sujetos.
    """)


    # Balancing dataset with undersampling
    class_0 = diabetes[diabetes['Outcome'] == 0]
    class_1 = diabetes[diabetes['Outcome'] == 1]
    class_0_reduced = class_0.sample(n=len(class_0) - 200, random_state=42)
    diabetes_balanced = pd.concat([class_0_reduced, class_1])

    # Show balanced class distribution
    st.subheader("Distribución de clases tras aplicar balanceo (undersampling)")
    balanced_class_distribution = diabetes_balanced['Outcome'].value_counts()
    st.bar_chart(balanced_class_distribution)

    
    # Tabla de desempeño
    st.subheader("Desempeño del modelo")
    st.write(results_df)

    
    st.markdown("### Cross Validation")

    st.markdown("""
    Para la validación se utilizó el método de **Stratified K-Fold**, que divide los datos en *k* conjuntos.  
    El modelo se entrena *k* veces, cambiando en cada iteración el subconjunto utilizado para evaluación.  
    Se usó una división del **80/20** para los datos de entrenamiento y prueba.
    """)

    st.markdown("### Modelos de Clasificación")

    st.markdown("""
    Se entrenaron **tres modelos de clasificación** para este proceso:

    - **Regresión Logística**: Estima la probabilidad de que un elemento pertenezca a una de dos clases (*diabético* o *no diabético*).  
    - **Árbol de Clasificación**: Separa los datos en *subsets* basados en el valor de características específicas, obteniendo al final grupos de elementos con características similares.  
    - **Random Forest**: Construye múltiples árboles de clasificación, donde cada árbol emite una predicción, y la clase más frecuente es la decisión final del modelo.  

    El algoritmo guarda los resultados obtenidos con cada modelo y muestra **el mejor resultado**.
    """)



    #Decision tree
    st.subheader("Resultado - Árbol de clasificación")
    # Acceder al árbol de decisión en el pipeline
    tree_model = best_model.named_steps['model']  # 'model' is the step name from the pipeline

    # Graficar el árbol de decisión
    fig, ax = plt.subplots(figsize=(12, 8))  # Define figure and axis
    plot_tree(tree_model, filled=True, feature_names=X.columns, rounded=True, ax=ax)  # Use ax
    st.pyplot(fig)  # Display in Streamlit

    # Confusion matrix
    st.subheader("Matriz de confusión del mejor modelo")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Show accuracy and recall
    st.subheader("Desempeño del modelo final")
    st.write(f"**Mejor modelo:** {model_name}")
    st.write(f"**Accuracy:** {test_accuracy:.4f}")
    st.write(f"**Recall:** {test_recall:.4f}")

    st.markdown("### Mejor Modelo y Resultados")

    st.markdown("""
    El mejor resultado obtenido fue con un **Árbol de Clasificación**, logrando una precisión de **0.8213**.

    En las hojas del árbol se observan varios valores altos de **entropía**, lo que indica que hay muestras mezcladas.  
    Esto puede reducir la certeza en la clasificación.  

    Sin embargo, también existen hojas con valores bajos o incluso **entropía = 0**, lo que sugiere nodos más puros.  
    En estos casos, el modelo puede clasificar nuevos datos con mayor confianza.

    ### Matriz de Confusión  
    La mayoría de los datos fueron correctamente clasificados, aunque se presentaron algunos errores.  
    En particular, hay **falsos positivos** (casos incorrectamente clasificados como diabéticos).  
    """)


#-------------------------------- Tab 3: Clustering --------------------------------
with tab3:
    
    dataset = diabetes
    
    # Class distribution
    st.subheader("Distribución de clases en Dataset original")
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
    distances = np.sort(distances[:, -1])  # Tomar la última distancia (k-ésimo vecino)

    # Crear imagen
    st.subheader("Gráfico para encontrar Epsilon (k-Distance)")
    fig, ax = plt.subplots()
    ax.plot(distances)
    ax.set_xlabel("Sorted Points")
    ax.set_ylabel(f"Distance to {min_samples}-th Nearest Neighbor")
    ax.set_title("k-Distance Graph for Finding Epsilon")

    # Mostrar gráfico en Streamlit
    st.pyplot(fig)

    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=1.4, min_samples=10)
    dbscan_labels = dbscan.fit_predict(data_scaled)

    #Plot DBSCAN
    st.subheader("Clusters identificados con DBSCAN")
    fig_dbscan = px.scatter(data_pca, 
                            x=data_pca[:,0], 
                            y=data_pca[:,1], 
                            color=dbscan_labels.astype(str),  # Convertir a string para evitar escalas numéricas en la leyenda
                            title="Clusters Identificados con DBSCAN",
                            labels={"PC1": "Componente Principal 1", "PC2": "Componente Principal 2", "color": "Cluster DBSCAN"},
                            opacity=0.7)  # Hacer los puntos semitransparentes para mejor visualización

    st.plotly_chart(fig_dbscan)

    if 'Outcome' in df.columns:
        df['DBSCAN_Cluster'] = dbscan_labels  # Agregar la columna de clusters al dataframe

        # Calcular distribución de Outcome por cluster
        outcome_distribution = df.groupby('DBSCAN_Cluster')['Outcome'].value_counts(normalize=True).unstack()

        # Mostrar tabla en Streamlit
        st.subheader("Distribución de Outcome por Clúster (DBSCAN)")
        st.write(outcome_distribution)  # Mostrar tabla en la app


    # Agrupar y calcular frecuencia y proporción
    df_count = df.groupby(["DBSCAN_Cluster", "Outcome"]).size().reset_index(name="Frecuencia")
    df_total = df.groupby("DBSCAN_Cluster").size().reset_index(name="Total")
    df_count = df_count.merge(df_total, on="DBSCAN_Cluster")
    df_count["Proporción"] = df_count["Frecuencia"] / df_count["Total"]

    # Convertir Outcome a string para que Plotly lo trate como categoría
    df_count["Outcome"] = df_count["Outcome"].astype(str)

    # Crear gráfico interactivo con Plotly
    fig = px.bar(df_count, 
                x="DBSCAN_Cluster", 
                y="Frecuencia", 
                color="Outcome",
                text=df_count["Proporción"].apply(lambda x: f"{x:.2%}"),  # Mostrar porcentaje
                barmode="group",  # Barras agrupadas (lado a lado)
                labels={"DBSCAN_Cluster": "Clúster DBSCAN", "Frecuencia": "Frecuencia", "Outcome": "Outcome"},
                title="Distribución de Outcome en Clusters DBSCAN")

    # Mostrar gráfico en Streamlit
    st.plotly_chart(fig)

    # Interpretación en texto
    st.write("* DBSCAN identificó un gran grupo de 'outliers' (Cluster -1) con un 45% de diabéticos. Esto significa que muchos puntos no se pudieron agrupar bien.")
    st.write("* El único cluster grande (Cluster 0) tiene un 50-50 de pacientes diabéticos y no diabéticos, lo que indica que DBSCAN no logró encontrar una separación clara entre los grupos.")
    st.write("No generó múltiples clusters útiles → Podría significar que los datos no tienen agrupaciones naturales bien definidas.")


    # Evaluación de K-Means con diferentes números de clusters
    k_values = range(2, 10)
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_scaled)
        silhouette = silhouette_score(data_scaled, labels)
        silhouette_scores.append(silhouette)

    # Mostrar gráfica en Streamlit
    st.subheader("Evaluación de K-Means con diferentes Clusters")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, silhouette_scores, marker='o', linestyle='-')
    ax.set_xlabel('Número de Clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Evaluación de K-Means')
    st.pyplot(fig)

    # Aplicar K-Means con el mejor número de clusters encontrado
    best_k = k_values[silhouette_scores.index(max(silhouette_scores))]
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(data_scaled)

    # Visualización de Clusters con PCA
    st.subheader(f"Clusters Identificados con K-Means (k={best_k})")
    fig_pca = px.scatter(
        x=data_pca[:, 0], 
        y=data_pca[:, 1], 
        color=kmeans_labels.astype(str),
        title=f"Clusters Identificados con K-Means (k={best_k})",
        labels={"color": "Cluster K-Means"}
    )
    st.plotly_chart(fig_pca)

    # Relación entre los clusters y la variable Outcome si está disponible
    if 'Outcome' in df.columns:
        df['KMeans_Cluster'] = kmeans_labels

        st.subheader("Distribución de Outcome por Clúster (K-Means)")
        df_kmeans_count = df.groupby(["KMeans_Cluster", "Outcome"]).size().reset_index(name="Frecuencia")
        df_kmeans_total = df.groupby("KMeans_Cluster").size().reset_index(name="Total")
        df_kmeans_count = df_kmeans_count.merge(df_kmeans_total, on="KMeans_Cluster")
        df_kmeans_count["Proporción"] = df_kmeans_count["Frecuencia"] / df_kmeans_count["Total"]
        df_kmeans_count["Outcome"] = df_kmeans_count["Outcome"].astype(str)

        # Crear gráfico de barras con proporciones
        fig_kmeans_dist = px.bar(
            df_kmeans_count, 
            x="KMeans_Cluster", 
            y="Frecuencia", 
            color="Outcome",
            text=df_kmeans_count["Proporción"].apply(lambda x: f"{x:.2%}"),
            barmode="group",
            labels={"KMeans_Cluster": "Clúster K-Means", "Frecuencia": "Frecuencia", "Outcome": "Outcome"},
            title="Distribución de Outcome en Clusters K-Means"
        )
        st.plotly_chart(fig_kmeans_dist)

    # Análisis de características por cluster
    st.subheader("Promedios de características por cluster (K-Means)")
    cluster_means = pd.DataFrame(data_scaled, columns=df.drop(columns=['Outcome', 'KMeans_Cluster', 'DBSCAN_Cluster'], errors='ignore').columns)
    cluster_means['KMeans_Cluster'] = kmeans_labels
    st.dataframe(cluster_means.groupby('KMeans_Cluster').mean())


    #Conclusiones
    st.subheader("Conclusiones")

    st.markdown("""
    **En conclusión:**
    * K-Means es mejor que DBSCAN en este caso porque logra formar grupos más interpretables de riesgo bajo, intermedio y alto de diabetes.
    * DBSCAN no encontró clusters útiles y clasificó muchos puntos como outliers, lo que sugiere que los datos no tienen una estructura de agrupación clara.

    Luego de obtener la distribución y el Silhouette Score de cada evaluación del algoritmo en k entre 2 y 9, se confirma visualmente que el valor más óptimo, en este caso, es k=3.

    Posterior a encontrar el mejor k, se aplica nuevamente K-Means, se visualiza con PCA y se evalúa nuevamente la relación entre Outcome y los Clusters, encontrando lo siguiente:

    * Los clusters están separados, sin embargo, hay solapamiento en algunas áreas. Adicionalmente, el cluster amarillo parece estar en el centro, conectando los otros dos clusters.

    ### **Proporción de diabéticos y no diabéticos por clusters:**
      - **Cluster 0:** 68.29% diabéticos (Outcome=1) → **Grupo de alto riesgo**.
      - **Cluster 1:** 60.00% diabéticos → **Grupo intermedio**.
      - **Cluster 2:** 37.21% diabéticos → **Grupo de bajo riesgo**.

    El clustering logró separar parcialmente los grupos de alto y bajo riesgo. Sin embargo, Cluster 1 y Cluster 2 están algo mezclados, lo que indica que algunos pacientes tienen características comunes.

    ### **Características Promedio por Cluster:**
    ✅ **Cluster 0 (alto riesgo de diabetes):** Contiene la mayor cantidad de embarazos (Pregnancies=0.92), mayor glucosa (Glucose=0.37) y mayor presión arterial (BloodPressure=0.39).  
    ✅ **Cluster 1 (intermedio):** Glucosa y BMI están en valores intermedios y la presión arterial es más baja que en los otros clusters.  
    ✅ **Cluster 2 (bajo riesgo de diabetes):** Contiene la menor cantidad de embarazos (Pregnancies=-0.58) y menores valores de glucosa y BMI.

    El clustering tiene sentido porque el grupo de alto riesgo tiene mayor glucosa y mayor cantidad de embarazos, que son factores de riesgo conocidos para la diabetes.
    El grupo de bajo riesgo tiene menores niveles en estas variables. El grupo intermedio tiene características mixtas.
    """)
