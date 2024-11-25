import streamlit as st
from typing import Sequence
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
import pandas as pd
from langchain.vectorstores import FAISS
import uuid

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.error("⛔ Accès refusé. Veuillez vous authentifier sur la page Presentation.")
    st.stop()  # Arrête l'exécution du script ici
# === Setup LLM ===
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=st.secrets["openai_api_key"])

# === Load Data ===
@st.cache_data
def load_metadata():
    return pd.read_csv("data/metadata_pages.csv")

df_metadata = load_metadata()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=st.secrets["openai_api_key"])

# Load FAISS index
@st.cache_resource
def load_faiss_index():
    """
    Load the FAISS index saved in the 'data/' directory.
    """
    vectorstore = FAISS.load_local(
        "data/faiss_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

vectorstore = load_faiss_index()


# Sidebar filter for entity selection
selected_groupe = st.sidebar.multiselect(
    "Filtre par groupe:",
    options=["All"] + sorted(df_metadata["Groupe"].unique()),
    default=["All"]
)

# Sidebar filter for entity selection
selected_entities = st.sidebar.multiselect(
    "Filtre par entité:",
    options=["All"] + sorted(df_metadata["Entity"].unique()),
    default=["All"]
)


# Construire le filtre multiple
filter_conditions = {}

# Ajouter un filtre pour les entités sélectionnées (si plusieurs entités)
if "All" not in selected_entities and selected_entities:
    filter = {
      "Entity" : list(selected_entities)
    }

# Ajouter un filtre pour les groupes sélectionnés (si plusieurs groupes)
if "All" not in selected_groupe and selected_groupe:
   filter = {
      "Groupe": list(selected_groupe)
   }

if ("All" not in selected_entities and selected_entities) or ("All" not in selected_groupe and selected_groupe):

  # Configurer le retriever avec le filtre multiple
  retriever = vectorstore.as_retriever(
      search_type="similarity",
      search_kwargs={
          "k": 10,  # Nombre de résultats à retourner
          "filter": filter
      }
  )
else : 
  # Configurer le retriever avec le filtre multiple
  retriever = vectorstore.as_retriever(
      search_type="similarity",
      search_kwargs={
          "k": 10,  # Nombre de résultats à retourner
      }
  )


# === Contextualize Questions ===
contextualize_q_system_prompt = (
    "En vous basant sur l'historique de la conversation et la dernière question posée par l'utilisateur, "
    "formulez une question autonome qui puisse être comprise sans se référer à cet historique. "
    "Ne répondez pas à la question, reformulez-la uniquement si nécessaire, ou renvoyez-la telle quelle."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# === Create Retrieval Chain ===
system_prompt = (
    "Vous êtes un assistant spécialisé dans les thématiques de la Quatrième Conférence Internationale sur le Financement du Développement (FFD4). "
    "Votre rôle est de fournir des réponses détaillées, claires et bien structurées, basées sur le contexte récupéré à partir des contributions officielles. "
    "Ces contributions incluent les priorités, recommandations et propositions politiques de divers acteurs "
    "(gouvernements, organisations internationales, société civile, etc.). "
    "Utilisez le contexte suivant pour répondre à la question de manière précise et complète. "
    "Si aucune information pertinente ne figure dans le contexte, indiquez clairement que vous ne pouvez pas répondre.\n\n"
    "{context}"
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# === Define Chat Workflow ===
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

def call_model(state: State):
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }

workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# === Streamlit UI ===
st.title("🔎 Discute avec les Contributions")
st.caption(
    "💡 Ce chatbot se souvient du contexte de vos questions précédentes pour fournir des réponses plus précises et pertinentes. "
    "Pour accéder aux sources utilisées dans les réponses, appuie sur la section '🔍 Sources' située sous chaque réponse."
)


# Handle chat history and responses
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

input_text = st.chat_input("Pose des questions sur les contributions, le rôle des BPD, etc.")
if input_text:
    state = {
        "input": input_text,
        "chat_history": st.session_state["chat_history"],
    }

    result = app.invoke(state,config=config)
       # Update session state with the response
    st.session_state["chat_history"] = result["chat_history"]

    # Display the question
    st.markdown(f"🧑 **Toi:** {input_text}")

    # Display the answer
    st.markdown(f"🤖 **Assistant:** {result['answer']}")

    # Display retrieved context in a well-formatted expander
    with st.expander("🔍 Sources",expanded=False):
        st.markdown("### **Documents extraits**")
        for idx, doc in enumerate(result["context"], start=1):
            metadata = doc.metadata
            content = doc.page_content
            st.markdown(f"#### **Document {idx}:**")
            st.markdown(f"- **Numéro de la page:** {metadata.get('Page_Number', 'N/A')}")
            st.markdown(f"- **Entité:** {metadata.get('Entity', 'N/A')}")
            st.markdown(f"- **Groupe:** {metadata.get('Groupe', 'N/A')}")
            st.markdown(f"- **Lien:** {metadata.get('Link', 'N/A')}")
            st.markdown("#### **Contexte:**")
            st.write(content)
