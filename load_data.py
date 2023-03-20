import streamlit as st
import time
import pandas as pd
import geopandas as gpd
from dashboard import dashboard
from data_map import data_map
st.set_page_config(layout="wide")
def mode(df):
    m = df.value_counts().head(1).index
    return m

def read_data():
    df = pd.read_csv("data full clean.csv")
    df_pivot = df.pivot_table(index = "area",
                                values = ["rent","rooms","year"],
                                aggfunc = {"rent":["mean","min","max",mode],"rooms":"count","year":["min","max"]})
    # Read the shapefile-formatted (.shp) file
    gdf = gpd.read_file('bezirksgrenzen.shp')
    list1 = gdf.index.values.tolist()
    list2 = ["Reinickendorf","Charlottenburg-Wilmersdorf",
            "Treptow-Köpenick","Pankow","Neukölln","Lichtenberg",
            "Marzahn-Hellersdorf","Spandau","Steglitz-Zehlendorf",
            "Mitte","Friedrichshain-Kreuzberg","Tempelhof-Schöneberg"]
    gdf = gdf.rename(index=dict(zip(list1,list2)))
    df1 = pd.concat([gdf,df_pivot], axis = 1)
    cols = ["geometry","max_rent","mean_rent","min_rent","mode_rent","available_apartments","newest_building","oldest_building"]
    df1.columns = cols
    cols = ["max_rent","mean_rent","min_rent","mode_rent"]
    for i in cols:
        rent = []
        for j in df1[i]:
            rent.append(round(j,0))
        df1[i] = rent
    return df,df1,df_pivot

def load_data():
    options = ['Load Dataset','Display Dashboard','Show Rent Prediction','Stop']
    choice = st.sidebar.selectbox("Menu",options, key = '1')
    if (choice == "Load Dataset"):
        st.write('<div style="text-align: center; color: yellow;"><h1>Berlin Apartment Rent Analysis</h1></div>', unsafe_allow_html=True)
        st.write(" ")
        st.write(" ")
        st.write(" ")
        readfile = st.button("Load Dataset")
        if readfile:
            df = pd.read_csv("data full clean.csv")
            df_pivot = df.pivot_table(index = "area",
                                        values = ["rent","rooms","year"],
                                        aggfunc = {"rent":["mean","min","max",mode],"rooms":"count","year":["min","max"]})
            # Read in the shapefile
            gdf = gpd.read_file('bezirksgrenzen.shp')
            list1 = gdf.index.values.tolist()
            list2 = ["Reinickendorf","Charlottenburg-Wilmersdorf",
                    "Treptow-Köpenick","Pankow","Neukölln","Lichtenberg",
                    "Marzahn-Hellersdorf","Spandau","Steglitz-Zehlendorf",
                    "Mitte","Friedrichshain-Kreuzberg","Tempelhof-Schöneberg"]
            gdf = gdf.rename(index=dict(zip(list1,list2)))
            df1 = pd.concat([gdf,df_pivot], axis = 1)
            cols = ["geometry","max_rent","mean_rent","min_rent","mode_rent","available_apartments","newest_building","oldest_building"]
            df1.columns = cols

            cols = ["max_rent","mean_rent","min_rent","mode_rent"]
            for i in cols:
                rent = []
                for j in df1[i]:
                    rent.append(round(j,0))
                df1[i] = rent
        
            st.write("Successfully loaded dataset.")
            time.sleep(1)
            st.write("Here is a preview of the data:")
            
        pass
    elif (choice == 'Display Dashboard'):
        
        df,df1,df_pivot = read_data()
        dashboard(df,df1)
        pass

    elif (choice == 'Show Rent Prediction'):

        st.write("prediction")
        pass
    else:
        st.stop()
load_data()

    
