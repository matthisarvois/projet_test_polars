import streamlit as st 
import polars as pl  
from sklearn.datasets import load_iris
import time
import numpy as np
import plotly.express as px


iris = load_iris(as_frame=True)
iris.frame.to_csv("iris.csv", index=False)
lf_iris = pl.read_csv("iris.csv")


def app() -> None:
    """
    This function creates the principal page of the 
    """
    st.title("Première page du streamlit 🌳")
    
    st.markdown("---")
    st.markdown("## Première possibilité de faire un graphiques avec les différents filtres.")
    
    st.info("Ici se trouvera un graphiques avec des filtres pour changer les différentes manière de visualiser les données")
    
    #Création des status des sessions
    if 'df_iris' not in st.session_state:
        st.session_state['df_iris'] = None
        st.session_state['chargement'] = None
    
    c1,c2 = st.columns(2)
    with c1  :
        import_data_button = st.button("Import de la basee de données iris", use_container_width=True)
    
    if import_data_button :

        #Cacul of the time of calcul
        start = time.time()
        lf = import_lazzy()
        st.session_state['df_iris'] = lf
        st.session_state['chargement'] = time.time() - start
    
    if st.session_state['df_iris'] is not None:
        df = st.session_state['df_iris'].collect()
        with c2:
            
            with st.expander("Cacher les information de la base de donnée", expanded=True):
                #render of the dimension of the database and the time of chargement.
                st.success(f"✅Données importées en {np.round(st.session_state['chargement'],2)} secondes avec succès")
                col1, col2 = st.columns(2)
                col1.metric("Nombre de lignes du dataframe",df.height)
                col2.metric("Nombre de colonnes du dataframe",df.width)
            
        
        st.markdown("## Séléction de variable")
        selected_column = st.selectbox("Sélectionner une colonne", list(df.columns))
        
        lf = st.session_state['df_iris']
        #creation of the figure
        lf_filtred = lf.select(selected_column)
        df_filtred = lf_filtred.collect()
        
        fig_distrib_var = px.scatter(df_filtred, 
                                    x = selected_column)
        with st.expander("Cacher le graphique",expanded=True):
            st.plotly_chart(fig_distrib_var)
        
        st.markdown("## Séleciton d'une autre variable pour regarder leurs corrélation")
        colonnes = list(df.columns)
        colonnes.remove(selected_column)
        other_column = st.selectbox("Sélectionner une autre colonne", colonnes)
        
        
        with st.expander(f"Affichage de {selected_column} par rapport à {other_column}",expanded=True) :
            df_temp = lf.select(other_column, selected_column).collect()
        
            target_value = st.select_slider("Séléctionner une valeurs pour la target", sorted(df['target'].unique()))
            
            lf_filtred_2var = lf.filter(pl.col("target")== target_value)
            lf_filtred_2var = lf.select(selected_column, other_column,)
            df_filtred_2var = lf_filtred_2var.collect()
            
            fig_distrib_2var = px.scatter(df_filtred_2var, 
                                      x = other_column,
                                      y = selected_column)
            st.plotly_chart(fig_distrib_2var)
            
        
        
        
            
    return None


@st.cache_data
def import_lazzy()-> pl.LazyFrame:
    """  
        __summary : Import the iris frame and creates as LazzyFrame of the data
        __arguments : Nothing
        __out : The Lazzyframe of the data
         
    """
    iris = load_iris(as_frame=True)
    iris.frame.to_csv("iris.csv", index=False)
    lf_iris = pl.scan_csv("iris.csv")
    
    return lf_iris
    
