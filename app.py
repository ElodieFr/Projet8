import mlflow
import mlflow.pyfunc
import pandas as pd
import streamlit as st
import os
import plotly.graph_objects as go
import chardet
import urllib.request

# Définition des chemins locaux
DATA_PATH = "data/"
MODEL_PATH = "models/model"
OPTIMAL_THRESHOLD = 0.636364

# Vérification de l'existence des fichiers
@st.cache_data()
def check_file_exists(file_path):
    return os.path.exists(file_path)

# Détection automatique de l'encodage
@st.cache_data(ttl=600)
def detect_encoding(file_path):
    if not check_file_exists(file_path):
        return "utf-8"
    
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read(100000))
    return result.get("encoding", "utf-8")

# Chargement des données CSV avec gestion d'encodage
@st.cache_data(ttl=600)
def load_csv_data(filename):
    file_path = os.path.join(DATA_PATH, filename)
    
    if not check_file_exists(file_path):
        st.error(f"❌ Fichier introuvable : {filename}")
        return None

    try:
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding, on_bad_lines="skip")
        return df
    except Exception as e:
        st.error(f"⚠️ Erreur lors du chargement {filename} : {str(e)}")
        return None

@st.cache_data(ttl=600)
def load_feature_names():
    file_path = os.path.join(DATA_PATH, "feature_names.txt")
    if check_file_exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    return []

@st.cache_resource()
def load_model():
    if check_file_exists(MODEL_PATH):
        try:
            return mlflow.sklearn.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"⚠️ Erreur lors du chargement du modèle : {str(e)}")
    return None

# Téléchargement automatique du fichier CSV si nécessaire
url = "https://raw.githubusercontent.com//ElodieFr/Projet8/master/data/application_test.csv"
destination = os.path.join("data", "application_test.csv")

if not os.path.exists(destination):
    os.makedirs("data", exist_ok=True)
    try:
        urllib.request.urlretrieve(url, destination)
        print("✅ Fichier téléchargé avec succès :", destination)
    except Exception as e:
        print(f"⚠️ Erreur lors du téléchargement du fichier : {str(e)}")

# Chargement des ressources
model = load_model()
feature_names = load_feature_names()
customer_data = load_csv_data("application_test.csv")
customer_data_description = load_csv_data("HomeCredit_columns_description.csv")

# Vérifier si les fichiers essentiels sont bien chargés
if customer_data is None:
    st.error("⚠️ Le fichier `application_test.csv` est introuvable ou corrompu.")
if customer_data_description is None:
    st.error("⚠️ Le fichier `HomeCredit_columns_description.csv` est introuvable ou corrompu.")
if model is None:
    st.error("⚠️ Le modèle est introuvable ou corrompu.")
if customer_data is None or model is None:
    st.stop()

# Prétraitement des données avant la prédiction
def preprocess_features(input_data):
    df = pd.DataFrame([input_data])

    # Vérifier et encoder les variables catégorielles
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes  # Convertir les catégories en nombres

    return df

# Fonction de prédiction sécurisée
def make_prediction(input_data, model, threshold):
    input_df = preprocess_features(input_data)

    # Vérifier les colonnes attendues et les comparer avec celles en entrée
    st.write("🚀 Colonnes attendues par le modèle:", feature_names)
    st.write("🚀 Colonnes reçues après preprocessing:", input_df.columns.tolist())
    st.write("🚀 Nombre de colonnes après preprocessing:", input_df.shape[1])

    # Vérifier que toutes les colonnes du modèle sont présentes dans l'input
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0  # Ajouter la colonne manquante avec une valeur par défaut

    # Réordonner les colonnes pour correspondre au modèle
    input_df = input_df[feature_names]

    st.write("✅ Nombre de colonnes après alignement:", input_df.shape[1])

    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[:, 1][0]
            return prob, "Refusé" if prob >= threshold else "Accepté"
        else:
            st.error("⚠️ Le modèle ne supporte pas `predict_proba`")
    except Exception as e:
        st.error(f"⚠️ Erreur lors de la prédiction : {str(e)}")
        return None, None

# Affichage d'une jauge avec Plotly
def gauge_chart(value, threshold):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Probabilité de défaut de paiement"},
        gauge={'axis': {'range': [0, 1]},
               'bar': {'color': "red" if value >= threshold else "green"},
               'steps': [
                   {'range': [0, threshold], 'color': "green"},
                   {'range': [threshold, 1], 'color': "red"}] }
    ))
    st.plotly_chart(fig)

# Interface principale
def main():
    st.title("📊 Credit Scoring Dashboard")
    st.header("🔍 Sélection du client")

    # Vérifier si customer_data contient bien la colonne SK_ID_CURR
    if 'SK_ID_CURR' not in customer_data.columns:
        st.error("⚠️ Erreur : La colonne `SK_ID_CURR` est absente des données.")
        st.stop()

    sk_id_curr = st.number_input("Entrez un SK_ID_CURR:",
                                 min_value=int(customer_data['SK_ID_CURR'].min()),
                                 max_value=int(customer_data['SK_ID_CURR'].max()))

    if sk_id_curr in customer_data['SK_ID_CURR'].values:
        client_data = customer_data[customer_data['SK_ID_CURR'] == sk_id_curr].iloc[0].to_dict()
        st.write("📋 **Informations du client :**")
        st.json(client_data)
        
        # Prédiction
        prob, label = make_prediction(client_data, model, OPTIMAL_THRESHOLD)
        if prob is not None:
            gauge_chart(prob, OPTIMAL_THRESHOLD)
            st.success(f"**Résultat de la prédiction : {label}**")
    else:
        st.warning("⚠️ Client non trouvé. Veuillez entrer un ID valide.")

    # Mode debug pour vérifier les fichiers
    if st.checkbox("🔍 Mode Debug"):
        st.write("Fichiers disponibles dans 'data/':", os.listdir(DATA_PATH) if check_file_exists(DATA_PATH) else "Dossier introuvable")
        st.write("Fichiers disponibles dans 'models/':", os.listdir("models/") if check_file_exists("models/") else "Dossier introuvable")
        st.write("Feature Names:", feature_names)
        st.write("Aperçu des données clients:", customer_data.head() if customer_data is not None else "Données introuvables")

if __name__ == "__main__":
    main()
