
# ================== IMPORTS & CONFIG ==================
import streamlit as st
st.set_page_config(layout="wide")
import pickle
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def preparer_input_data(donnees_dict):
    import pandas as pd

    df = pd.DataFrame([donnees_dict])
    df = df[colonnes_features]

    # Utiliser les colonnes originales du scaler pour éviter les erreurs
    cols_numeriques = scaler.feature_names_in_
    df[cols_numeriques] = scaler.transform(df[cols_numeriques])

    return df
# ================== pour historique de prédiction==================
colonnes_attendues = [
    "date", "fournisseur", "nom_fournisseur",
    "article", "description_article", "quantite",
    "date_commande", "statut"
]

# ================== CSS DESIGN ==================
st.markdown("""
    <style>
        /* Supprime l'en-tête Streamlit */
        header[data-testid="stHeader"] {
            height: 0rem;
            visibility: hidden;
        }

        /* Réduit espace au-dessus du contenu */
        .block-container {
            padding-top: 0rem !important;
        }

        /* Supprime marges autour des logos */
        img {
            margin-top: 0px !important;
            margin-bottom: 0px !important;
            padding-top: 0px !important;
        }

        main {
            padding-top: 0rem !important;
        }

    /* 🔹 Elargir la sidebar */
    section[data-testid="stSidebar"] {
        width: 380px !important;
        background: linear-gradient(180deg, #1E3C72 0%, #2A5298 100%);
        padding: 40px 25px;
        height: 100vh;
        border-top-right-radius: 40px;
        border-bottom-right-radius: 40px;
        color: white;
    }


    /* 🔹 Style des boutons radio (texte + icône) */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        font-size: 24px !important;
        font-weight: 700 !important;
        color: white !important;
        padding: 10px 18px;
        margin-bottom: 12px;
        border-radius: 12px;
        display: flex !important;
        align-items: center;
        gap: 10px;
        transition: background 0.3s ease;
    }

    /* 🔹 Sélection et hover */
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {
        background-color: rgba(255, 255, 255, 0.2);
        font-weight: 900 !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(255, 255, 255, 0.1);
        cursor: pointer;
   
    
    }

    /* 🔹 Markdown personnalisé (ex : titres comme "Navigation") */
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p:not(.menu-title) {
    font-size: 26px !important;
    font-weight: bold !important;
    color: #FFFFFF !important;
    padding-left: 8px;
    }

            
            
    /* ✅ Trait uniquement sous "Navigation" */
    .menu-title {
    font-size: 40px !important;
    font-weight: bold !important;
    color: #FFFFFF !important;
    padding-left: 8px;
    margin-bottom: 20px;
    border-bottom: 2px solid rgba(255, 255, 255, 0.3);
    padding-bottom: 10px;
    }

    </style>
     
        
""", unsafe_allow_html=True)


# ================== LOGOS GLOBAL (TOUTES PAGES) ==================
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.image("assets/logo_entreprise.png", width=300)
with col3:
    st.image("assets/logo_ecole.png", width=300)

st.markdown("<br>", unsafe_allow_html=True)





# ================== SESSION STATE ==================
if "page" not in st.session_state:
    st.session_state["page"] = "🏠 Accueil"

if "pred_button_clicked" not in st.session_state:
    st.session_state["pred_button_clicked"] = False


# ================== SIDEBAR ==================

st.sidebar.markdown('<p class="menu-title">📁 Navigation</p>', unsafe_allow_html=True)

page_selection = st.sidebar.radio("", [
    "🏠 Accueil",
    "📊 Prédiction",
    "🔎 Exploration",
    "📈 Performances",
    "📅 Export",
    "📙 Historique des Prédictions"
], key="navigation_radio")

# Réinitialisation si changement de page
if page_selection != st.session_state["page"]:
    st.session_state["pred_button_clicked"] = False
    st.session_state["page"] = page_selection

# Page active
page = st.session_state["page"]

# ================== CHARGEMENT FICHIERS ==================
model = joblib.load("model_gb.pkl")
scaler = joblib.load("scaler.pkl")
with open("encoders_cat.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("label_encoder_target.pkl", "rb") as f:
    le_target = pickle.load(f)
with open("feature_columns.pkl", "rb") as f:
    colonnes_features = pickle.load(f)

df_hist = pd.read_csv("historique_streamlit.csv", dtype={"code_fournisseur": str, "article_code": str})
df_hist['creation_commande'] = pd.to_datetime(df_hist['creation_commande'])
df_hist['code_fournisseur'] = df_hist['code_fournisseur'].str.strip()
df_hist['article_code'] = df_hist['article_code'].str.strip()

# ================== PAGE ACCUEIL ==================
if page == "🏠 Accueil":
 
# 🎯 Titre + Sous-titre centré avec effet professionnel
  st.markdown("""
    <div style="text-align: center; margin-top: -140px; margin-bottom: 30px;">
        <h1 style="font-size: 65px; color: #003366; font-weight: 800;">
            🚚 <span style="color: ##1E3C72;">SupplyPilot</span>
        </h1>
        <h3 style="font-size: 40px; color: #1E3C72;">
            Application de Prédiction & <b>Suivi Logistique</b>
        </h3>
        <p style="font-size: 28px; color: #444; line-height: 1.6; max-width: 900px; margin: auto;">
            Une solution intelligente pour piloter la chaîne d’approvisionnement<br>
            et anticiper les risques de retard ou de non-livraison, en entreprise.
        </p>
        <p style="font-size: 18px; color: #555; margin-top: 10px;">
            <b>Technos :</b> Python · Streamlit · Machine Learning (Gradient Boosting) · Feature Engineering
        </p>
    </div>
""", unsafe_allow_html=True)


  
    # 📸 Image Supply Chain + 📊 Stats
 # 📸 Bloc image + 6 stats centrées verticalement
  col_left, col_right = st.columns([1.2, 2])

  with col_left:
    st.image("assets/supplychain.png", use_column_width=True)

  with col_right:
    st.markdown("<div style='margin-top:80px;'>", unsafe_allow_html=True)

    # Première ligne : statuts
    stat1, stat2, stat3 = st.columns(3)
    with stat1:
        st.markdown(f"""
            <div style="background-color:#e8f5e9; padding:15px; border-radius:12px; text-align:center;">
                🟢 <b>À temps</b><br>
                <span style="font-size:22px;">{(df_hist['statut_livraison'] == 'À temps').mean()*100:.1f}%</span>
            </div>
        """, unsafe_allow_html=True)

    with stat2:
        st.markdown(f"""
            <div style="background-color:#fff3e0; padding:15px; border-radius:12px; text-align:center;">
                🟠 <b>En retard</b><br>
                <span style="font-size:22px;">{(df_hist['statut_livraison'] == 'En retard').mean()*100:.1f}%</span>
            </div>
        """, unsafe_allow_html=True)

    with stat3:
        st.markdown(f"""
            <div style="background-color:#fce4ec; padding:15px; border-radius:12px; text-align:center;">
                🔴 <b>Non livrée</b><br>
                <span style="font-size:22px;">{(df_hist['statut_livraison'] == 'Non livrée').mean()*100:.1f}%</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Deuxième ligne : chiffres clés
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div style="background-color:#eef6ff; padding:15px; border-radius:12px; text-align:center;">
                📦 <b>Commandes</b><br>
                <span style="font-size:20px;">{len(df_hist)}</span>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background-color:#eef6ff; padding:15px; border-radius:12px; text-align:center;">
                🏭 <b>Fournisseurs</b><br>
                <span style="font-size:20px;">{df_hist['code_fournisseur'].nunique()}</span>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div style="background-color:#eef6ff; padding:15px; border-radius:12px; text-align:center;">
                📄 <b>Articles</b><br>
                <span style="font-size:20px;">{df_hist['article_code'].nunique()}</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


    with st.expander("🧐 À propos du modèle utilisé"):
        st.info("""
        Modèle Gradient Boosting Classifier entraîné sur des données internes.
        Il exploite des variables comme l'ancienneté du fournisseur, les taux de retard et de non-livraison,
        les caractéristiques temporelles (mois, année), la quantité commandée, etc.
        """)

    
# ========================== PAGE PRÉDICTION ==============================
if page == "📊 Prédiction":
  
  # ======================== Initialisation dictionnaires ========================
    fournisseurs = dict(zip(df_hist['code_fournisseur'], df_hist['nom_fournisseur']))
    articles_dict = dict(zip(df_hist['article_code'], df_hist['article_description']))

    # ======================== CSS ========================
    st.markdown("""
        <style>
            html, body, [class*="st-"] {
                font-size: 22px !important;
                font-family: "Segoe UI", sans-serif !important;
            }
            .main-title {
                text-align: center;
                font-size: 45px !important;
                font-weight: 900 !important;
                margin-top: -100px !important;
                margin-bottom: 30px !important;
                color: #1E3C72 !important;
            }
            div[data-testid="stHorizontalBlock"] label {
                font-size: 30px !important;
                font-weight: 800 !important;
                color: #1E3C72 !important;
            }
            div[role="radiogroup"] label {
                font-size: 26px !important;
                font-weight: 700 !important;
            }
            div[data-testid*="Input"] label,
            div[data-testid*="Selectbox"] label,
            div[data-testid*="Radio"] label {
                font-size: 26px !important;
                font-weight: bold !important;
                color: #1A1A1A !important;
            }
            input, select, textarea, div[data-baseweb="input"] {
                font-size: 22px !important;
                font-weight: 600 !important;
            }
            .stButton > button {
                font-size: 24px !important;
                font-weight: bold !important;
                padding: 0.8em 1.8em !important;
                background-color: #1f77b4 !important;
                color: white !important;
                border-radius: 8px !important;
                border: none !important;
                margin-top: 20px !important;
            }
            /* Uniformiser le style des champs texte personnalisés */
            input[type="text"] {
                font-size: 20px !important;
                font-weight: bold !important;
                color: #1E3C72 !important;
                background-color: #f8f9fa !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-title'>📊 Prédiction Statut de Livraison</div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:30px; font-weight:800; color:#1E3C72;'>🔧 Mode de saisie :</p>", unsafe_allow_html=True)
    mode = st.radio("", ["📂 Depuis historique", "✍️ Saisie manuelle","⬆️ Importer un fichier"], horizontal=True)
    st.session_state["input_mode"] = mode

    if "last_mode" not in st.session_state:
        st.session_state["last_mode"] = mode
    elif st.session_state["last_mode"] != mode:
        st.session_state["pred_button_clicked"] = False
        st.session_state["last_mode"] = mode
     
# ========================== mode import  ==============================#
    if mode == "⬆️ Importer un fichier":
        st.subheader("📤 Importer un fichier CSV")
        fichier = st.file_uploader("Choisissez un fichier .csv", type=["csv"])

        if fichier is not None:
            df_import = pd.read_csv(fichier)
            st.write("Aperçu du fichier importé :", df_import.head())

            try:
                # 1. Convertir les dates
                df_import['creation_commande'] = pd.to_datetime(df_import['creation_commande'])
                df_import['date_attendue'] = pd.to_datetime(df_import['date_attendue'], errors='coerce')
                df_import['date_livraison'] = pd.to_datetime(df_import['date_livraison'], errors='coerce')

                # 2. Colonnes temporelles
                df_import['mois'] = df_import['creation_commande'].dt.month
                df_import['annee'] = df_import['creation_commande'].dt.year
                df_import['delai_prevu'] = (df_import['date_attendue'] - df_import['creation_commande']).dt.days

                # 3. Ancienneté fournisseur depuis l’historique
                df_hist['creation_commande'] = pd.to_datetime(df_hist['creation_commande'])
                first_orders = df_hist.groupby('code_fournisseur')['creation_commande'].min().reset_index()
                first_orders.columns = ['code_fournisseur', 'first_order']
                df_import = df_import.merge(first_orders, on='code_fournisseur', how='left')
                df_import['anciennete'] = (df_import['creation_commande'] - df_import['first_order']).dt.days.fillna(0)
                df_import['anciennete_fournisseur_log'] = np.log1p(df_import['anciennete'])

                # 4. Taux retard / non livrée par fournisseur (historique)
                df_hist = df_hist.sort_values(['code_fournisseur', 'creation_commande'])
                df_hist['is_retard'] = (df_hist['statut_livraison'] == 'En retard').astype(int)
                df_hist['is_nonlivree'] = (df_hist['statut_livraison'] == 'Non livrée').astype(int)
                df_hist['cumul_retard'] = df_hist.groupby('code_fournisseur')['is_retard'].cumsum().shift().fillna(0)
                df_hist['cumul_nonlivree'] = df_hist.groupby('code_fournisseur')['is_nonlivree'].cumsum().shift().fillna(0)
                df_hist['cumul_total'] = df_hist.groupby('code_fournisseur').cumcount()

                taux_df = df_hist.groupby('code_fournisseur')[['cumul_retard', 'cumul_nonlivree', 'cumul_total']].last().reset_index()
                taux_df['taux_retard'] = (taux_df['cumul_retard'] / taux_df['cumul_total'].replace(0, np.nan)).fillna(0)
                taux_df['taux_nonlivree'] = (taux_df['cumul_nonlivree'] / taux_df['cumul_total'].replace(0, np.nan)).fillna(0)
                df_import = df_import.merge(taux_df[['code_fournisseur', 'taux_retard', 'taux_nonlivree']], on='code_fournisseur', how='left')

                # 5. Taux retard / non livrée par fournisseur + article
                df_hist['fourn_art'] = df_hist['code_fournisseur'] + "_" + df_hist['article_code']
                df_import['fourn_art'] = df_import['code_fournisseur'] + "_" + df_import['article_code']

                df_hist['cumul_retard_fa'] = df_hist.groupby('fourn_art')['is_retard'].cumsum().shift().fillna(0)
                df_hist['cumul_nonlivree_fa'] = df_hist.groupby('fourn_art')['is_nonlivree'].cumsum().shift().fillna(0)
                df_hist['cumul_total_fa'] = df_hist.groupby('fourn_art').cumcount()

                taux_fa_df = df_hist.groupby('fourn_art')[['cumul_retard_fa', 'cumul_nonlivree_fa', 'cumul_total_fa']].last().reset_index()
                taux_fa_df['taux_retard_fa'] = (taux_fa_df['cumul_retard_fa'] / taux_fa_df['cumul_total_fa'].replace(0, np.nan)).fillna(0)
                taux_fa_df['taux_nonlivree_fa'] = (taux_fa_df['cumul_nonlivree_fa'] / taux_fa_df['cumul_total_fa'].replace(0, np.nan)).fillna(0)
                df_import = df_import.merge(taux_fa_df[['fourn_art', 'taux_retard_fa', 'taux_nonlivree_fa']], on='fourn_art', how='left')

                # 6. Sauvegarde des valeurs originales pour encodage
                df_import['code_fournisseur_original'] = df_import['code_fournisseur']
                df_import['article_code_original'] = df_import['article_code']

                # 7. Encodage
                df_import['code_fournisseur'] = df_import['code_fournisseur_original'].apply(
                    lambda x: encoders['code_fournisseur'].transform([x])[0] if x in encoders['code_fournisseur'].classes_ else 999
                )
                df_import['article_code'] = df_import['article_code_original'].apply(
                    lambda x: encoders['article_code'].transform([x])[0] if x in encoders['article_code'].classes_ else 999
                )
                df_import['accuse_reception'] = df_import['accuse_reception'].apply(
                    lambda x: encoders['accuse_reception'].transform([x])[0] if x in encoders['accuse_reception'].classes_ else 999
                )

                # 8. Normalisation
                colonnes_numeriques = scaler.feature_names_in_
                df_import[colonnes_numeriques] = scaler.transform(df_import[colonnes_numeriques])

                # 9. Charger la liste des colonnes du modèle
                with open("feature_columns.pkl", "rb") as f:
                    colonnes_model = pickle.load(f)

                # 10. Sélectionner les colonnes dans le bon ordre
                X = df_import[colonnes_model]

                # 11. Prédiction
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)



                labels = ['À temps', 'En retard', 'Non livrée']
                df_import['statut_prédit'] = [labels[i] for i in y_pred]
                df_import['proba_à_temps'] = y_proba[:, 0]
                df_import['proba_en_retard'] = y_proba[:, 1]
                df_import['proba_non_livrée'] = y_proba[:, 2]

                # 12. Affichage des résultats
                st.success("✅ Prédictions terminées.")
                st.dataframe(df_import[[
                    'code_fournisseur_original', 'nom_fournisseur',
                    'article_code_original', 'article_description',
                    'statut_prédit', 'proba_à_temps', 'proba_en_retard', 'proba_non_livrée'
                ]])

                # ⚡️ Le bouton 
                csv = df_import.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

                st.markdown("""
                <style>
                div.stDownloadButton > button {
                    background-color: #0a58ca;
                    color: white;
                    border-radius: 8px;
                    padding: 8px 24px;
                    font-weight: bold;
                }
                div.stDownloadButton > button:hover {
                    background-color: #084298;
                    color: white;
                }
                </style>
            """, unsafe_allow_html=True)

                  
                st.download_button(
                    label="📥 Télécharger les résultats",
                    data=csv,
                    file_name="resultats_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Une erreur est survenue : {e}")

                

# ========================== mode saisie ==============================
    col1, col2 = st.columns(2)

    with col1:
        if mode == "✍️ Saisie manuelle":
            code_fournisseur = st.text_input("Code fournisseur :")
            nom_fournisseur = st.text_input("Nom fournisseur :")

            article_code = st.text_input("Code article :")
            description_article = st.text_input("Description article :")

            quantite_pieces = st.number_input("Quantité", min_value=1, value=100)
            date_creation_commande = st.date_input("Date de création de la commande", value=datetime.today())
            anciennete = st.number_input("Ancienneté (jours)", value=120)
            taux_retard = st.slider("Taux de retard", 0.0, 1.0, 0.2)
            taux_nonlivree = st.slider("Taux de non livraison", 0.0, 1.0, 0.1)
            accuse_reception = st.radio("Accusé réception ?", ["Oui", "Non"], horizontal=True)
            accuse_reception_value = 1 if accuse_reception == "Oui" else 0
            mois = st.slider("Mois", 1, 12, datetime.today().month)
            annee = st.slider("Année", 2020, datetime.today().year, datetime.today().year)
            delai_prevu = st.number_input("Délai prévu", value=10)
            anciennete_log = np.log1p(anciennete)

            if st.button("📈 Prédire", key="predict_manuelle"):
                code_fournisseur_enc = (
                    encoders['code_fournisseur'].transform([code_fournisseur])[0]
                    if code_fournisseur in encoders['code_fournisseur'].classes_
                    else 999
                )
                article_code_enc = (
                    encoders['article_code'].transform([article_code])[0]
                    if article_code in encoders['article_code'].classes_
                    else 999
                )


                st.session_state["pred_button_clicked"] = True
                st.session_state["input_data"] = {
                    'anciennete_fournisseur_log': anciennete_log,
                    'accuse_reception': accuse_reception_value,
                    'mois': mois,
                    'delai_prevu': delai_prevu,
                    'annee': annee,
                    'taux_retard': taux_retard,
                    'taux_nonlivree': taux_nonlivree,
                    'taux_retard_fa': taux_retard,
                    'taux_nonlivree_fa': taux_nonlivree,
                    'article_code': article_code_enc,
                    'code_fournisseur': code_fournisseur_enc,
                    'quantite_pieces': quantite_pieces,
                    'fournisseur': code_fournisseur,
                    'article': article_code,
                    'nom_fournisseur': nom_fournisseur,
                    'description_article': description_article
                }
# ========================== mode historique  ==============================

        elif mode == "📂 Depuis historique":
            code_fournisseur = st.selectbox("Fournisseur :", sorted(df_hist['code_fournisseur'].unique()))
            st.markdown(f"📃 **Nom fournisseur :** `{fournisseurs.get(code_fournisseur, 'Non trouvé')}`")

            article_list = df_hist[df_hist['code_fournisseur'] == code_fournisseur]['article_code'].unique()
            article_code = st.selectbox("Article :", article_list)
            st.markdown(f"🧳 **Description article :** `{articles_dict.get(article_code, 'Non trouvé')}`")

            quantite_pieces = st.number_input("Quantité", min_value=1, value=100)
            date_creation_commande = st.date_input("Date de création", value=datetime.today())
            accuse_reception = st.radio("Accusé réception ?", ["Oui", "Non"], horizontal=True)
            accuse_reception_value = 1 if accuse_reception == "Oui" else 0

            df_filtre = df_hist[(df_hist['code_fournisseur'] == code_fournisseur) &
                                (df_hist['article_code'] == article_code) &
                                (df_hist['creation_commande'] < pd.to_datetime(date_creation_commande))]

            if df_filtre.empty:
                st.warning("⚠️ Aucun historique trouvé avant cette date.")
            else:
                taux_retard = (df_filtre['statut_livraison'] == 'En retard').mean()
                taux_nonlivree = (df_filtre['statut_livraison'] == 'Non livrée').mean()
                anciennete = (pd.to_datetime(date_creation_commande) - df_filtre['creation_commande'].min()).days
                anciennete_log = np.log1p(anciennete)
                mois = pd.to_datetime(date_creation_commande).month
                annee = pd.to_datetime(date_creation_commande).year
                # Cherche la dernière commande (plus récente) dans l'historique filtré
                ligne_ref = df_filtre.sort_values(by="creation_commande", ascending=False).iloc[0]

                # S'assure que les colonnes sont bien en format datetime
                ligne_ref['date_attendue'] = pd.to_datetime(ligne_ref['date_attendue'])
                ligne_ref['creation_commande'] = pd.to_datetime(ligne_ref['creation_commande'])

                # Calcule le délai prévu réel
                delai_prevu = (ligne_ref['date_attendue'] - ligne_ref['creation_commande']).days

                # ✅ Affiche dans l'application
                st.markdown(f"⏳ **Délai de production (historique) :** `{delai_prevu} jours`")

                if st.button("📈 Prédire", key="predict_historique"):

                    code_fournisseur_enc = (
                        encoders['code_fournisseur'].transform([code_fournisseur])[0]
                        if code_fournisseur in encoders['code_fournisseur'].classes_
                        else 999
                    )

                    article_code_enc = (
                        encoders['article_code'].transform([article_code])[0]
                        if article_code in encoders['article_code'].classes_
                        else 999
                    )


                    st.session_state["pred_button_clicked"] = True
                    st.session_state["input_data"] = {
                        'anciennete_fournisseur_log': anciennete_log,
                        'accuse_reception': accuse_reception_value,
                        'mois': mois,
                        'delai_prevu': delai_prevu,
                        'annee': annee,
                        'taux_retard': taux_retard,
                        'taux_nonlivree': taux_nonlivree,
                        'taux_retard_fa': taux_retard,
                        'taux_nonlivree_fa': taux_nonlivree,
                        'article_code': article_code_enc,
                        'code_fournisseur': code_fournisseur_enc,
                        'quantite_pieces': quantite_pieces,
                        'fournisseur': code_fournisseur,
                        'article': article_code
                    }

                

    with col2:
        if st.session_state.get("pred_button_clicked") and st.session_state.get("input_mode") == mode:
            input_data = st.session_state["input_data"]
            features_df = preparer_input_data(input_data)

            prediction = model.predict(features_df)[0]
            proba = model.predict_proba(features_df)[0]
            labels = ['À temps', 'En retard', 'Non livrée']
            statut = labels[prediction]

            
            now = datetime.now().strftime("%Y-%m-%d %H:%M")

            # 🔹 Création d’une ligne d’historique

            if mode == "✍️ Saisie manuelle":
                nom_fournisseur = input_data.get("nom_fournisseur", "Inconnu")
                description_article = input_data.get("description_article", "Inconnu")
            else:
                nom_fournisseur = fournisseurs.get(input_data["fournisseur"], "Inconnu")
                description_article = articles_dict.get(input_data["article"], "Inconnu")


            ligne = {
                "date": now,
                "fournisseur": str(input_data["fournisseur"]),
                "nom_fournisseur": nom_fournisseur,
                "article": str(input_data["article"]),
                "description_article": description_article,
                "quantite": input_data["quantite_pieces"],
                "date_commande": date_creation_commande.strftime("%d/%m/%Y"),
                "statut": statut
            }

            # 🔹 Ajout au fichier CSV
            df_ligne = pd.DataFrame([ligne])
            df_ligne.to_csv("historique_predictions.csv", sep=";", mode="a", index=False,
                            header=not os.path.exists("historique_predictions.csv"))

            couleurs = {
                "À temps": "#2ECC71",
                "En retard": "#E67E22",
                "Non livrée": "#E74C3C"
            }

            st.markdown(f"""
                <div style='
                    background-color: {couleurs[statut]}22;
                    border-left: 8px solid {couleurs[statut]};
                    padding: 20px;
                    border-radius: 12px;
                    font-size: 26px;
                    font-weight: bold;
                    color: {couleurs[statut]};
                    text-align: center;
                    margin-top: 20px;
                '>
                    🚚 Statut prédit : {statut}
                </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 Détails des probabilités", expanded=True):
                fig, ax = plt.subplots(figsize=(6, 5))
                
                # Affiche chaque barre avec la couleur associée à son label
                bars = ax.bar(labels, proba, color=[couleurs[l] for l in labels])
                
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("Probabilité")
                ax.set_title("Distribution des probabilités")

                for bar, val in zip(bars, proba):
                    ax.annotate(f'{val:.2%}',
                                xy=(bar.get_x() + bar.get_width() / 2, val),
                                xytext=(0, 8),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=10, weight='bold')

                st.pyplot(fig)



# ================================Historique des Prédictions=============================

if page == "📙 Historique des Prédictions": 
    st.markdown("""
        <style>
            body {
                zoom: 1.2;
            }

            .main-title {
                text-align: center;
                font-size: 45px;
                font-weight: 900;
                margin-top: -140px;
                margin-bottom: 30px;
                color: #1E3C72;
            }

            .section-title {
                font-size: 30px !important;
                font-weight: 900 !important;
                color: #1f1f1f !important;
                margin-top: 10px !important;
                margin-bottom: 20px !important;
            }

            /* Texte dans le menu déroulant */
            div[data-baseweb="select"] {
                font-size: 24px !important;
            }

            div[role="option"] {
                font-size: 24px !important;
            }
                
                /* 🎨 Style des boutons bleus */
        .stButton > button, .stDownloadButton > button {
            background-color: #1f77b4 !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.6em 1.5em !important;
            font-size: 20px !important;
            font-weight: bold !important;
            border: none !important;
            margin-top: 10px !important;
        }
    
        </style>
    """, unsafe_allow_html=True)

    # Titre principal
    st.markdown("<div class='main-title'>📙 Historique des Prédictions</div>", unsafe_allow_html=True)

    # Chargement du CSV
    if os.path.exists("historique_predictions.csv"):
        df_predictions = pd.read_csv("historique_predictions.csv", sep=";")

        colonnes_manquantes = [col for col in colonnes_attendues if col not in df_predictions.columns]

        if colonnes_manquantes:
            st.warning(f"⚠️ Colonnes manquantes dans le fichier : {colonnes_manquantes}. Le fichier a été réinitialisé.")
            df_predictions = pd.DataFrame(columns=colonnes_attendues)
            df_predictions.to_csv("historique_predictions.csv", sep=";", index=False)
    else:
        df_predictions = pd.DataFrame(columns=colonnes_attendues)
        df_predictions.to_csv("historique_predictions.csv", sep=";", index=False)

    # 📤 Affichage ou message si vide
    if df_predictions.empty:
        st.info("L'historique est vide.")
    else:
        # 📑 Filtres
        st.markdown("<div class='section-title'>📑 Filtres</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<p style='font-size:26px; font-weight:bold; margin-bottom:5px;'>Fournisseur</p>", unsafe_allow_html=True)
            filt_fournisseur = st.selectbox("", ["Tous"] + sorted(df_predictions["fournisseur"].astype(str).unique()))

        with col2:
            st.markdown("<p style='font-size:26px; font-weight:bold; margin-bottom:5px;'>Article</p>", unsafe_allow_html=True)
            filt_article = st.selectbox("", ["Tous"] + sorted(df_predictions["article"].astype(str).unique()))

        # 🧼 Application des filtres
        df_filtered = df_predictions.copy()
        if filt_fournisseur != "Tous":
            df_filtered = df_filtered[df_filtered["fournisseur"].astype(str) == filt_fournisseur]
        if filt_article != "Tous":
            df_filtered = df_filtered[df_filtered["article"].astype(str) == filt_article]

        # 📊 Affichage
        if df_filtered.empty:
            st.warning("Aucun résultat pour ces filtres.")
        else:
            st.markdown("<div class='section-title'>📊 Résultats</div>", unsafe_allow_html=True)
            st.dataframe(df_filtered, use_container_width=True)

        # 📦 Boutons en bas
        col1, spacer, col2 = st.columns([1, 0.05, 1])

        with col1:
            csv_data = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Télécharger en CSV", csv_data, "historique_filtré.csv", "text/csv")

        with col2:
            if st.button("🗑️ Vider l’historique"):
                df_predictions = pd.DataFrame(columns=colonnes_attendues)
                df_predictions.to_csv("historique_predictions.csv", sep=";", index=False)
                st.success("✅ L'historique a été vidé.")
                st.experimental_rerun()
# ================================


# 🔎 Exploration


import matplotlib.pyplot as plt
import io
from PIL import Image

if page == "🔎 Exploration":
    st.markdown("""
        <div style="text-align: center; margin-top: -140px; margin-bottom: 30px;">
            <h1 style="font-size: 55px; color: #1E3C72; font-weight: 900;">
                🔎 Exploration du Dataset
            </h1>
        </div>

        <style>
            table {
                font-size: 18px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.write("## Aperçu des premières lignes :")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    st.table(df_hist.head(10))

    st.write("## Répartition des statuts de livraison")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # ➕ Données
    statut_counts = df_hist['statut_livraison'].value_counts()

    
        # 📏 Taille plus grande
    fig, ax = plt.subplots(figsize=(12, 8.6))  # 🔹 plus grand
    bars = ax.bar(statut_counts.index, statut_counts.values, color='#1E90FF')

    # Titres et axes
    ax.set_title("Répartition des statuts de livraison", fontsize=14)
    ax.set_xlabel("Statut", fontsize=12)
    ax.set_ylabel("Nombre de commandes", fontsize=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    plt.tight_layout()

    # 🖼️ Convertir en image avec meilleure qualité
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)  # ✅ meilleure qualité
    buf.seek(0)
    img = Image.open(buf)

    # ✅ Affichage sans déformation
    st.image(img, use_column_width=False)

    # 📅 Saisonnalité des retards par mois

    # 📅 Saisonnalité des retards par mois
    st.write("## 📅 Saisonnalité des retards par mois")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)



    df_hist['mois'] = df_hist['creation_commande'].dt.month
    df_mois = df_hist.groupby(['mois', 'statut_livraison']).size().unstack().fillna(0)
    df_mois_percent = df_mois.div(df_mois.sum(axis=1), axis=0)

    fig_mois, ax_mois = plt.subplots(figsize=(12, 8.6))  # taille équilibrée
    df_mois_percent.plot(kind='bar', stacked=True, ax=ax_mois, colormap='Set3')
    ax_mois.set_title("Répartition des statuts par mois", fontsize=15)
    ax_mois.set_xlabel("Mois", fontsize=12)
    ax_mois.set_ylabel("Proportion", fontsize=12)
    ax_mois.legend(title="Statut", fontsize=10)
    ax_mois.tick_params(axis='x', labelsize=10)
    ax_mois.tick_params(axis='y', labelsize=10)
    plt.tight_layout()

    buf_mois = io.BytesIO()
    fig_mois.savefig(buf_mois, format="png", dpi=130)
    buf_mois.seek(0)
    st.image(Image.open(buf_mois), use_column_width=False)

   

    # 🏆 Top fournisseurs fiables
    st.write("## 🏆 Fournisseurs les plus fiables vs les moins fiables")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    seuil_commande = 30
    stats_fourn = df_hist.groupby('nom_fournisseur')['statut_livraison'].value_counts(normalize=True).unstack().fillna(0)
    stats_fourn = stats_fourn[df_hist['nom_fournisseur'].value_counts() > seuil_commande]

    # ✅ Meilleurs fournisseurs
    stats_fourn_sorted = stats_fourn.sort_values(by='À temps', ascending=False).head(10)
    fig_best, ax_best = plt.subplots(figsize=(12, 8.6))
    stats_fourn_sorted['À temps'].plot(kind='bar', ax=ax_best, color='green')
    ax_best.set_title("Top 10 des fournisseurs les plus fiables", fontsize=14)
    ax_best.set_ylabel("Taux de livraisons à temps", fontsize=12)
    ax_best.tick_params(axis='x', labelsize=10)
    ax_best.tick_params(axis='y', labelsize=10)
    plt.tight_layout()

    buf_best = io.BytesIO()
    fig_best.savefig(buf_best, format="png", dpi=130)
    buf_best.seek(0)
    st.image(Image.open(buf_best), use_column_width=False)

    # ❌ Moins bons fournisseurs
    stats_fourn_worst = stats_fourn.sort_values(by='À temps', ascending=True).head(10)
    fig_worst, ax_worst = plt.subplots(figsize=(12, 8.6))
    stats_fourn_worst['À temps'].plot(kind='bar', ax=ax_worst, color='red')
    ax_worst.set_title("Top 10 des fournisseurs les moins fiables", fontsize=14)
    ax_worst.set_ylabel("Taux de livraisons à temps", fontsize=12)
    ax_worst.tick_params(axis='x', labelsize=10)
    ax_worst.tick_params(axis='y', labelsize=10)
    plt.tight_layout()

    buf_worst = io.BytesIO()
    fig_worst.savefig(buf_worst, format="png", dpi=130)
    buf_worst.seek(0)
    st.image(Image.open(buf_worst), use_column_width=False)
# ================================ 📈 Performances=========================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 📈 Performances
if page == "📈 Performances":


    # 🔹 Titre centré hors du bloc
    st.markdown("""
        <h1 style='text-align: center; font-size: 55px; font-weight: 900; color: #1E3C72; margin-top: -140px; margin-bottom: 30px;'>
            📈 Performances du modèle
        </h1>
    """, unsafe_allow_html=True)

    # 🔹 Bloc de performance avec fond
    st.markdown("""
        <div style="
            background-color: #f5f8fa;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #d6e2e9;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 25px;
        ">
            <p style='font-size: 24px; line-height: 1.6;'>
                Voici les performances du modèle <b>Gradient Boosting</b> lors des phases de test :
            </p>
            <ul style='font-size: 18px; line-height: 2.0; margin-left: 25px;'>
                <li><span style="font-size: 24px;">🎯 <b>Accuracy test :</b> 81.7%</span></li>
                <li><span style="font-size: 24px;">📊 <b>F1-score macro :</b> 0.82</span></li>
                <li><span style="font-size: 24px;">❄️ <b>Modèle final utilisé :</b> <span style="color:#0074A2;"><b>Gradient Boosting Classifier</b></span></span></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)


    target_names = ['En retard', 'Non livrée', 'À temps']

    if os.path.exists("C:/Users/khali/Desktop/streamlit/y_true_preds.csv"):
        df_perf = pd.read_csv("C:/Users/khali/Desktop/streamlit/y_true_preds.csv", sep=";")


        # Mapper les labels numériques en texte
        mapping = {0: "En retard", 1: "Non livrée", 2: "À temps"}
        df_perf["y_true"] = df_perf["y_true"].map(mapping)
        df_perf["y_pred"] = df_perf["y_pred"].map(mapping)

        # Matrice de confusion  
        
        cm = confusion_matrix(df_perf["y_true"], df_perf["y_pred"], labels=target_names)
        fig_cm, ax_cm = plt.subplots(figsize=(5.2, 3.6))  # ⬅️ Taille plus grande

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(ax=ax_cm, cmap="Blues", values_format="d", colorbar=False)

        # Supprimer les titres
        ax_cm.set_xlabel('')
        ax_cm.set_ylabel('')

        # Taille des ticks et labels
        ax_cm.tick_params(labelsize=10)
        for label in ax_cm.get_xticklabels() + ax_cm.get_yticklabels():
            label.set_fontsize(10)

        # Taille des chiffres internes
        for text in disp.text_.ravel():
            text.set_fontsize(10)

        plt.tight_layout(pad=0.4)

        # Affichage à gauche
        col_gauche, _ = st.columns([2, 4])
        with col_gauche:
            st.markdown("""
        <h2 style='font-size:35px;'>🧩 Matrice de confusion</h2>
        """, unsafe_allow_html=True)

            st.pyplot(fig_cm)
        # Ajout d'un espace
        st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)


    # 1. Générer le rapport
    from sklearn.metrics import classification_report

    report = classification_report(
        df_perf["y_true"], df_perf["y_pred"],
        labels=target_names,
        output_dict=True
    )
    df_report = pd.DataFrame(report).transpose().round(2)

    # 2. Convertir en HTML (sans escape)
    html_table = df_report.to_html(classes="rapport-table", border=0, escape=False)

    # 3. Titre stylisé
    st.markdown("""
    <h3 style='font-size: 35 px;'>📋 Rapport par classe (précision, rappel, F1-score)</h3>
    """, unsafe_allow_html=True)

    # 4. Affichage du tableau HTML avec style
    st.markdown("""
        <style>
            .rapport-table {
                font-size: 20px;
                width: 750px;
                margin-left: 0;
                border-collapse: collapse;
            }
            .rapport-table th, .rapport-table td {
                border: 1px solid #ccc;
                padding: 8px 12px;
                text-align: center;
            }
            .rapport-table thead {
                background-color: #f0f2f6;
            }
        </style>
    """, unsafe_allow_html=True)

    # 5. Injecter la table HTML
    st.markdown(html_table, unsafe_allow_html=True)



# ========================== PAGE EXPORT ==============================


elif page == "📅 Export":
    st.markdown("""
        <style>
            body {
                zoom: 1.2;
            }

            .main-title {
                text-align: center;
                font-size: 45px;
                font-weight: 900;
                margin-top: -140px;
                margin-bottom: 30px;
                color: #1E3C72;
            }

            /* 🟦 Bouton bleu */
            .stDownloadButton > button {
                background-color: #1f77b4 !important;
                color: white !important;
                font-size: 24px !important;
                padding: 1em 2.5em !important;
                font-weight: 600 !important;
                border-radius: 8px !important;
                border: none !important;
                margin-top: 10px !important;
            }

            div[data-testid="stMarkdownContainer"] > p {
                font-size: 22px !important;
                line-height: 1.8;
            }
        </style>
    """, unsafe_allow_html=True)

    # ✅ Titre cohérent avec Historique
    st.markdown("<h1 class='main-title'>📥 Exportation du Dataset</h1>", unsafe_allow_html=True)

    # 🔹 Texte explicatif
    st.markdown("<p style='font-size: 26px !important;'>Vous pouvez télécharger l'historique complet sous forme de fichier CSV.</p>", unsafe_allow_html=True)

    # 🔹 Bouton téléchargement
    if not df_hist.empty:
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger le dataset complet (CSV)",
            data=csv,
            file_name='historique_streamlit.csv',
            mime='text/csv',
        )
    else:
        st.warning("🚫 Le dataset est vide. Aucune donnée à exporter.")




# ================================
# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; font-size: 22px;'>
        © 2025 – Projet PFE Ingénierie des Données - EHTP | Réalisé par : <b>Sabrine Chiboub</b>
    </div>
""", unsafe_allow_html=True)
