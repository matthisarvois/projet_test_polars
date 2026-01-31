import streamlit as st 
import polars as pl  
from src.frontend.pages import page_1



def main() -> None:
    """  __summary: 
    """
    st.set_page_config(page_title="Premier prototype d'application entreprise",
                       page_icon="🌳",
                       layout= "wide")
    
    #On répertorie les différentes pages de notre applications
    PAGES = {
        "Première page 💻":page_1
    }
    
    #On créer la sidebar qui va contenir toutes les potentielles pages
    with  st.sidebar.title("Navigation") :
        selection_page = st.selectbox("Choisir une page",list(PAGES.keys()))
    
    page = PAGES[selection_page]
    page.app()
    return None


if __name__=="__main__":
    main()