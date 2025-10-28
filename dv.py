# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from PIL import Image
import matplotlib.pylab as plb
from st_aggrid import AgGrid, GridOptionsBuilder
from pathlib import Path
from som_revised import SelfOrganizingMap
from xlsx import XLSX
import logging


# logging.basicConfig(
#                 format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
#                 datefmt="%d-%m-%Y %H:%M:%S",
#                 level=logging.WARNING,
#                 filename='deep_cluster.log'
#                 )

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
def get_selected_rows_df(df,check: bool):
    # Create a GridOptionsBuilder instance
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(editable=False)  # Make all columns editable
    gb.configure_selection('multiple', use_checkbox=check)  # Enable checkbox selection
    grid_options = gb.build()
    grid_response = AgGrid(df, gridOptions=grid_options, editable=False)
    selected_rows = grid_response['selected_rows']
    # st.write(selected_rows)
    # st.write(type(selected_rows))
    if type(selected_rows) == pd.DataFrame:
        return selected_rows
        
# Set the theme to dark gray
st.set_page_config(page_title="Deep Veritas", page_icon="🧊", layout="wide", initial_sidebar_state="expanded")
# Load the custom CSS
load_css('styles.css')

# Title and subtitle
st.subheader("Deep Veritas, deep clustering for finding outliers")
st.text("Data Science Team, Specialist Business, Kantar India")

# Sidebar for file upload and download
st.sidebar.text("Control Panel")

def upload_and_convert(uploaded_file):
    # Upload file     
    if uploaded_file is not None:
        # Check the file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type!")
            return None
        
        st.success("File uploaded and converted successfully!")
        return df
    else:
        st.warning("Please upload a file.")
        return None

if st.sidebar.checkbox("Upload the dataset"):
    data_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx"])
    if data_file is not None:
        df = upload_and_convert(data_file)
        vars = []
        # st.write(f"Details of the uploaded dataset {data_file.name}:") 
        features = df.columns.to_list()
        if st.sidebar.checkbox("Select Relevant Features"):       
            vars = st.multiselect("Select relevant features to build the distance matrix",features)
        if st.sidebar.checkbox("Unselect Irrelevant Features"):
            vars_to_drop = st.multiselect("Select variables to drop",features)
            if len(vars_to_drop) > 0:
                vars = list(set(features) - set(vars_to_drop))
        if len(vars) > 0:
            X = df[vars]
            if st.sidebar.checkbox("Custom Selection of cases"):
                SX = get_selected_rows_df(X,check=True)
                X = SX if SX is not None else X 
            try:           
                data,X = SelfOrganizingMap.prepare_data_advanced(X)
                logging.info(f"dataset has been prepared and scaled successfully with dim {data.shape}")
            except Exception as err:
                st.warning(f"could not prepare the data: {err}")
            if st.sidebar.checkbox("Dimension of the model data"):
                st.info(f"Dimension of the Input Data: {X.shape}")
                get_selected_rows_df(data,check=False)
            st.info(f"{X.shape}")
            try:
                som = SelfOrganizingMap(10,10,X.shape[1],X,sigma=1,learning_rate=0.5)
                logger = logging.getLogger("deep_cluster")
                som.train(X,num_iterations=200)
                st.info("The model got trained successfully!")  
                if st.sidebar.checkbox("Retrain the model"):
                    lr = st.sidebar.number_input("learning rate:",value=0.5,max_value=1.0,step=0.1)
                    iterations = st.sidebar.number_input("Iterations:",value=200,step=50)
                    som.retrain(X,10,10,X.shape[1],learning_rate=lr,num_iterations=iterations)              
                if st.sidebar.checkbox("U-Matrix"):   
                    col1, col2 = st.columns(2)             
                    img_file = som.u_matrix()
                    img = Image.open(f"{img_file}")  
                    col1.image(img, caption=f"U-Matrix (Unified Distance Matrix)", use_container_width=True)
            except Exception as err:
                st.warning(f"U-Matrix failed: {err}")                
            if st.sidebar.checkbox("Range of Qunatization Errors"):
                outliers, qe = som.get_outlier_3sigma(X)
                st.info("Range of Quantization Errors:")
                st.info(f"Min {qe.min():.4f}, max {qe.max():.4f}")
                st.info(f"Outliers(3-σ): {outliers}")
            if st.sidebar.checkbox("Outliers based off Threshold"):
                value = qe.mean()+2.54*qe.std()
                # Initialize session state for the number input
                if 'threshold' not in st.session_state:
                    st.session_state.threshold = 0.0                 
                threshold = st.sidebar.number_input("Threshold",value=st.session_state.threshold, step=0.1)
                st.session_state.threshold = threshold
                if st.session_state.threshold > 0:
                    outliers_th = som.get_outliers_threshold(X,threshold=st.session_state.threshold)[0]
                    st.info(f"Outliers based off threshold of {threshold:.3f}\n{outliers_th}")
                if st.sidebar.checkbox("Save the list of Outliers"):
                    try:
                        xlsx = XLSX("outliers.xlsx")
                        tdf1 = df.loc[outliers_th]
                        tdf2 = df.loc[outliers]
                        xlsx.write_to_ws(tdf1, f"threshold", "Tab1")
                        xlsx.write_to_ws(tdf2,"Outliers (3-σ)","Tab2")
                        st.info("lists of outliers have been saved to outliers.xlsx")
                        xlsx.add_header_footer(header="Deep Cluster",footer="Data Science Team, Specialist Business, Kantar India")
                        xlsx.save_excel()
                                        
                        with open('outliers.xlsx', 'rb') as f:
                            st.sidebar.download_button('Download Outliers.xlsx', f, file_name='outliers.xlsx')
                        logging.info("Lists of outliers have been successfully saved and downloaded")
                    except Exception as err:
                        # pass
                        logger.error(f"Outlier lists couldn't be saved and downloaded: {err}")
        if st.sidebar.checkbox("Download Log"):
            with open('deep_cluster.log', 'rb') as f:
                st.sidebar.download_button('deep_cluster.log', f, file_name='deep_cluster.log')
        
                
            
         
     


