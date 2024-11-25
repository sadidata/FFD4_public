import hmac
import streamlit as st

# Simuler des secrets (normalement, ils sont stockés dans `st.secrets`)
USER_CREDENTIALS = {
    "AFD": {"password": "AFD_FFD4"},
}

st.set_page_config(
   layout="wide"
)
# Fonction pour vérifier l'authentification
def check_credentials():
    """Vérifie le nom d'utilisateur, le mot de passe et la clé API."""

    def credentials_entered():
        """Valide les identifiants entrés par l'utilisateur."""
        username = st.session_state.get("username", "")
        password = st.session_state.get("password", "")

        if username in USER_CREDENTIALS:
            user_data = USER_CREDENTIALS[username]
            if (
                hmac.compare_digest(password, user_data["password"])
            ):
                # Authentification réussie
                st.session_state["authenticated"] = True
                # Supprimer les informations sensibles des entrées
                del st.session_state["username"]
                del st.session_state["password"]
                return
        # Si les identifiants sont incorrects
        st.session_state["authenticated"] = False

    # Si l'utilisateur est déjà authentifié, retourner True
    if st.session_state.get("authenticated", False):
        return True

    # Interface d'entrée pour les identifiants
    st.title("🔐 Authentification requise")
    st.text_input("Nom d'utilisateur", key="username")
    st.text_input("Mot de passe", type="password", key="password")
    st.button("Se connecter", on_click=credentials_entered)

    # Message d'erreur si les identifiants sont incorrects
    if "authenticated" in st.session_state and not st.session_state["authenticated"]:
        st.error("😕 Nom d'utilisateur, mot de passe")
    return False

# Appel de la fonction pour gérer l'authentification
if not check_credentials():
    st.stop()  # Bloque l'exécution si l'utilisateur n'est pas authentifié

  
# --- Contenu de la Présentation ---
st.title("🌍 Analyse des Contributions FFD4")  # Titre principal

st.markdown(
    """

    #### Contexte général 📝
    - Dans le cadre du processus sur le **financement du développement** aboutissant à la Conférence de Séville en juin 2025 (FFD4),
      de nombreux acteurs (États membres 🌐, organisations internationales 🤝, et organisations de la société civile 🌱)
      ont partagé leurs contributions écrites auprès de **UNDESA**.

    #### **Objectifs du projet 🚀**
    1. **Analyser** environ **259 contributions** pour identifier les priorités de l’AFD et du FICS.
    2. Détecter les **mentions des BPD** (Banques Publiques de Développement) ainsi que leurs recommandations clés.
    4. Évaluer la part des objectifs de développement durable dans les contributions.
    3. Permettre une **lecture globale et rapide** grâce à des outils d’intelligence artificielle 🤖, pour extraire automatiquement les informations pertinentes et les structurer.

    ---

    #### **Description des onglets 🌟**

    ##### 1️⃣ **Chatbot Contributions**
    Cet onglet permettra d’interagir directement avec les contributions 📄. Grâce à des outils d’intelligence artificielle (IA), vous pourrez :
    - **Poser des questions précises** sur le contenu des contributions (par exemple, "Quels États membres mentionnent les BPD ?").
    - **Explorer et naviguer** parmi les documents de manière interactive et intuitive.
    - Gagner du temps en obtenant des réponses instantanées sans parcourir manuellement les documents.

    ##### 2️⃣ **Analyse des ODDs**
    Dans cet onglet, nous nous concentrerons sur les **Objectifs de Développement Durable (ODDs)** 🎯 :
    - Identifier le degré d'alignement des contributions sur les différents ODDs.
    - Repérer les priorités émergentes pour les acteurs clés et leur cohérence avec les objectifs globaux.
    - Fournir une synthèse claire sur les recommandations associées aux ODDs, facilitant les discussions stratégiques.

    ##### 3️⃣ **BDP et Financement**
    Cet onglet explorera le rôle des **Banques Publiques de Développement (BPD)** 🏦  :
    - Analyser les contributions mentionnant les BPD et leurs propositions spécifiques.
    - Identifier les recommandations clés pour renforcer l'impact des BPD dans le financement du développement.
    - Fournir une vue globale des suggestions de financement durable et innovant.

    ##### 4️⃣ Actions d'Addis-Abeba  
    Cet onglet est dédié à l’analyse des **Actions prioritaires de l’Agenda d’Addis-Abeba** :  
    - Étudier les contributions selon les **sept domaines d’action clés** (ressources publiques, commerce, dette, etc.).  
    - Fournir une vue d’ensemble des engagements et propositions pour le financement du développement.  

    """
)

st.markdown("---")

st.markdown(
    """
    **Développé par :**
    **INN - Agence Française de Développement**
    
    **Contact :** [Abdulaziz Sadi-Cherif](mailto:sadi-cherifa.ext@afd.fr)
    """
  )
