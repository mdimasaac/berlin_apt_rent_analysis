import streamlit as st
import time
import pandas as pd
import geopandas as gpd
from data_map import data_map
from prediction import prediction
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
    return df,df1

def dashboard():
    st.write('<div style="text-align: center; color: yellow;"><h1>Berlin Apartment Rent Analysis</h1></div>', unsafe_allow_html=True)
    st.write(" ")
    st.write(" ")
    st.write(" ")

    options = ['Display Dashboard','Show Rent Prediction','Stop']
    choice = st.sidebar.selectbox("Menu",options, key = '1')
    if (choice == "Display Dashboard"):      
        df,df1 = read_data()
        total_available = df.shape[0]
        cheapest_rent = df["rent"].min()
        district_cheapest_rent = df["area"][df["rent"]==df["rent"].min()].values[0]
        most_available_district = df1[df1["available_apartments"]==df1["available_apartments"].max()].index.values.tolist()[0]
        most_available_total = df1["available_apartments"].max()
        barrier_free_apartments = df[(df["barrier_free"]!=0) | (df["wheelchair_friendly"]!=0)].shape[0]
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.write('<div style="text-align: center;"><h5>Total Available<br>Apartments:<br></h5></div>', unsafe_allow_html=True)
            st.write('<div style="text-align: center;"><h1>'+str(total_available)+'</h1></div>', unsafe_allow_html=True)
            st.write('<div style="text-align: center;"><h6>Apartments<br></h6></div>', unsafe_allow_html=True)

        with col2:
            st.write('<div style="text-align: center;"><h5>Cheapest<br>Rent:<br></h5></div>', unsafe_allow_html=True)
            st.write('<div style="text-align: center;"><h1>'+str(cheapest_rent)+' €</h1></div>', unsafe_allow_html=True)
            st.write('<div style="text-align: center;"><h6>in '+str(district_cheapest_rent)+'</h6></div>', unsafe_allow_html=True)

        with col3:
            st.write('<div style="text-align: center;"><h5>Most Available<br>District:</h5></div>', unsafe_allow_html=True)
            st.write('<div style="text-align: center;"><h1>'+str(most_available_district)+'</h1></div>', unsafe_allow_html=True)
            st.write('<div style="text-align: center;"><h6>with '+str(most_available_total)+' apartments</h6></div>', unsafe_allow_html=True)

        with col4:
            st.write('<div style="text-align: center;"><h5>Apartments for People with<br>Health Conditions:</h5></div>', unsafe_allow_html=True)
            st.write('<div style="text-align: center;"><h1>'+str(barrier_free_apartments)+'</h1></div>', unsafe_allow_html=True)
            st.write('<div style="text-align: center;"><h6>Apartments</h6></div>', unsafe_allow_html=True)
        
        st.write("_________")
        tab1,tab2,tab3 = st.tabs(["Distribution by Rent Range","Display per District","Choropleth Map"])
        with tab1:
            col1,col2,col3 = st.columns([1.5,2,1.5])
            with col1:
                st.write(" ")
                st.write(" ")
                st.write(" ")
                filtered = df
                r = []
                a = []
                if st.checkbox("Filter Data by District"):
                    selected_area = st.selectbox("Select District", options=df['area'].unique())
                    a.append(selected_area)
                    st.write("Or Select Multiple Districts")
                    col31,col32 = st.columns(2)
                    with col31:
                        a00 = st.checkbox(filtered["area"].unique()[0])
                        if a00:
                            try:
                                a.remove(selected_area)
                            except:
                                pass
                            a.append(filtered["area"].unique()[0])
                        a01 = st.checkbox(filtered["area"].unique()[1])
                        if a01:
                            try:
                                a.remove(selected_area)
                            except:
                                pass
                            a.append(filtered["area"].unique()[1])
                        a02 = st.checkbox(filtered["area"].unique()[2])
                        if a02:
                            try:
                                a.remove(selected_area)
                            except:
                                pass
                            a.append(filtered["area"].unique()[2])
                        a03 = st.checkbox(filtered["area"].unique()[3])
                        if a03:
                            try:
                                a.remove(selected_area)
                            except:
                                pass
                            a.append(filtered["area"].unique()[3])
                        a04 = st.checkbox(filtered["area"].unique()[4])
                        if a04:
                            try:
                                a.remove(selected_area)
                            except:
                                pass
                            a.append(filtered["area"].unique()[4])
                        a05 = st.checkbox(filtered["area"].unique()[5])
                        if a05:
                            try:
                                a.remove(selected_area)
                            except:
                                pass
                            a.append(filtered["area"].unique()[5])
                    with col32:
                        a06 = st.checkbox(filtered["area"].unique()[6])
                        if a06:
                            try:
                                a.remove(selected_area)
                            except:
                                pass
                            a.append(filtered["area"].unique()[6])
                        a07 = st.checkbox(filtered["area"].unique()[7])
                        if a07:
                            try:
                                a.remove(selected_area)
                            except:
                                pass
                            a.append(filtered["area"].unique()[7])
                    
                        a08 = st.checkbox(filtered["area"].unique()[8])
                        if a08:
                            try:
                                a.remove(selected_area)
                            except:
                                pass
                            a.append(filtered["area"].unique()[8])
                        a09 = st.checkbox(filtered["area"].unique()[9])
                        if a09:
                            try:
                                a.remove(selected_area)
                            except:
                                pass
                            a.append(filtered["area"].unique()[9])
                        a10 = st.checkbox(filtered["area"].unique()[10])
                        if a10:
                            try:
                                a.remove(selected_area)
                            except:
                                pass
                            a.append(filtered["area"].unique()[10])
                        a11 = st.checkbox(filtered["area"].unique()[11])
                        if a11:
                            try:
                                a.remove(selected_area)
                            except:
                                pass
                            a.append(filtered["area"].unique()[11])    
                else:
                    selected_area = a = df["area"]
                st.write("_________")
                if st.checkbox("Filter Data by Number of Rooms"):
                    col11,col12,col13 = st.columns(3)
                    with col11:
                        r01 = st.checkbox("1 room") # BOOLEAN
                        if r01:
                            r.append(1.0)
                        r15 = st.checkbox("1.5 rooms")
                        if r15:
                            r.append(1.5)
                        r02 = st.checkbox("2 rooms")
                        if r02:
                            r.append(2.0)
                        r25 = st.checkbox("2.5 rooms")
                    with col12:
                        if r25:
                            r.append(2.5)
                        r03 = st.checkbox("3 rooms")
                        if r03:
                            r.append(3.0)
                        r35 = st.checkbox("3.5 rooms")
                        if r35:
                            r.append(3.5)
                        r04 = st.checkbox("4 rooms")
                        if r04:
                            r.append(4.0)
                        r05 = st.checkbox("5 rooms")
                    with col13:
                        if r05:
                            r.append(5.0)
                        r06 = st.checkbox("6 rooms")
                        if r06:
                            r.append(6.0)
                        r07 = st.checkbox("7 rooms")
                        if r07:
                            r.append(7.0)
                        r08 = st.checkbox("8 rooms")
                        if r08:
                            r.append(8.0)
                        r09 = st.checkbox("9 rooms")
                        if r09:
                            r.append(9.0)
                        if r01==r15==r02==r25==r03==r35==r04==r05==r06==r07==r08==r09==False:
                            r=df["rooms"]
                else:
                    r = df["rooms"]   
                filtered = ((df[(df['area'].isin(a)) & (df['rooms'].isin(r))] ))

            with col2:
                import plotly.graph_objs as go
                import plotly.colors
                from streamlit_plotly_events import plotly_events
                import numpy as np
                st.write(" ")
                st.write(" ")   
                st.write('<div style="text-align: center;"><h3>Displaying rent distribution <br>by rent range.</h3></div>', unsafe_allow_html=True)
                st.write('<div style="text-align: center;"><h6>Click on the chart area to select the rent range.</h6></div>', unsafe_allow_html=True)

                fig = go.Figure(go.Pie(
                    labels = filtered.groupby(["rent_range"]).size().reset_index(name="counts")["rent_range"],
                    values = filtered.groupby(["rent_range"]).size().reset_index(name="counts")["counts"],hovertemplate = "%{label}<br>%{percent} <extra></extra>",
                    hole=.7,marker_colors=plotly.colors.sequential.Viridis,
                    marker_line_color='white',
                    marker_line_width=1.25,
                    pull=[0.025,0.025,0.025,0.025,0.2,0.025],
                    textinfo="label+percent"
                ))
                style = """
                <style>
                .chart-wrapper {
                    display: flex;
                    justify-content: center;
                }
                </style>
                """
                st.write(style, unsafe_allow_html=True)
                fig.update_layout(
                    template="plotly_dark",
                    annotations=[dict(text='Rent Range<br>in Euro (€)', font_size=12, showarrow=False)]
                )
            
                fig.update_layout(
                    autosize=True,  # Automatically adjust size to fit entire layout
                    margin=dict(l=0, r=0, t=0, b=0),)  # Remove margin around chart
                fig.update_layout(showlegend=False)
                fig.update_layout(
                font=dict(
                size=16,))

                selected = plotly_events(fig)
                if len(selected) != 0:
                    ind = list(selected[0].keys())[1]
                    index = selected[0][ind]
                else:
                    ind = 0
                    index = 0

            with col3:
                st.write(" ")
                st.write(" ")
                if len(filtered.index) != 0:
                    rentrange = filtered.groupby(["rent_range"]).size().reset_index(name="counts")["rent_range"][index]
                    st.write('<div style="text-align: center;"><h3>Showing apartments with rent range from '+rentrange+'.</h3></div>', unsafe_allow_html=True)
                    st.write(" ")
                    st.write(" ")
                    st.write(" ")
                    st.write(" ")

                    col41,col42,col43 = st.columns(3)
                    with col41:
                        more = st.checkbox("Show more")
                    with col42:
                        sort_asc = st.button("Sort: asc")
                    with col43:
                        sort_desc = st.button("Sort: desc")
                    df_show = filtered[["rooms","rent","area","link"]][filtered["rent_range"]==rentrange]
                    if more:
                        if sort_asc:
                            sort_desc = False
                            st.dataframe(df_show.sort_values(by = "rent", ascending = 1),use_container_width=True)
                        elif sort_desc:
                            sort_asc = False
                            st.dataframe(df_show.sort_values(by = "rent", ascending = 0),use_container_width=True)
                        else:
                            st.dataframe(df_show,use_container_width=True)
                    elif sort_asc:
                        sort_desc = False
                        st.dataframe(df_show.sort_values(by = "rent", ascending = 1).head(3),use_container_width=True)
                    elif sort_desc:
                        sort_asc = False
                        st.dataframe(df_show.sort_values(by = "rent", ascending = 0).head(3),use_container_width=True)
                    else:
                        st.dataframe(df_show.head(3),use_container_width=True)
        
        with tab2:
            col21,col22,col23 = st.columns([1,1.5,1.5])

            data = {"District":df1.index.values.tolist(),"Average Rent":df1["mean_rent"].tolist(),"Available Apartments":df1["available_apartments"]}
            data = pd.DataFrame(data)
            data_sortrent = data.sort_values(by = "Average Rent", ascending = 1).reset_index(drop=1)

            with col21:
                r2 = []
                st.write(" ")
                st.write(" ")
                st.write(" ")
                st.write(" ")
                st.write(" ")
                st.write(" ")
                st.write(" ")
                st.write(" ")
                if st.checkbox("Filter by Number of Rooms"):
                    col11,col12 = st.columns(2)
                    with col11:
                        r201 = st.checkbox("1 room ") # BOOLEAN
                        if r201:
                            r2.append(1.0)
                        r215 = st.checkbox("1.5 rooms ")
                        if r215:
                            r2.append(1.5)
                        r202 = st.checkbox("2 rooms ")
                        if r202:
                            r2.append(2.0)
                        r225 = st.checkbox("2.5 rooms ")
                    
                        if r225:
                            r2.append(2.5)
                        r203 = st.checkbox("3 rooms ")
                        if r203:
                            r2.append(3.0)
                        r235 = st.checkbox("3.5 rooms ")
                        if r235:
                            r2.append(3.5)
                    with col12:
                        r204 = st.checkbox("4 rooms ")
                        if r204:
                            r2.append(4.0)
                        r205 = st.checkbox("5 rooms ")
                    
                        if r205:
                            r2.append(5.0)
                        r206 = st.checkbox("6 rooms ")
                        if r206:
                            r2.append(6.0)
                        r207 = st.checkbox("7 rooms ")
                        if r207:
                            r2.append(7.0)
                        r208 = st.checkbox("8 rooms ")
                        if r208:
                            r2.append(8.0)
                        r209 = st.checkbox("9 rooms ")
                        if r209:
                            r2.append(9.0)
                        if r201==r215==r202==r225==r203==r235==r204==r205==r206==r207==r208==r209==False:
                            r2=df["rooms"]
                else:
                    r2 = df["rooms"].unique()
                
                filtered2 = ((df[df['rooms'].isin(r2)]))

            with col22:
                
                st.write('<div style="text-align: center;"><h2>Average Monthly Rent<br>per District (in €)</h2></div>', unsafe_allow_html=True)
                st.write(" ")
                x1 = filtered2.groupby(["area"]).mean().reset_index()["rent"].round(2).sort_values(ascending = 1).reset_index(drop=1)
                y1 = filtered2.groupby(["area"]).mean().reset_index()["area"]
                
                fig1 = go.Figure([go.Bar(
                                        x = x1,
                                        y = y1,
                                        orientation='h',
                width = 0.5,
                text=x1.round(2),
                textposition='outside',
                marker=dict(color=x1, colorscale='viridis', line=dict(width=1.25, color='white')),
                textfont=dict(size=14))])
                fig1.update_layout(
                autosize=True,  # Automatically adjust size to fit entire layout
                margin=dict(l=0, r=2, t=0, b=0),
                xaxis=dict(range=[0, 1.9*max(x1)]))
                fig1.update_xaxes(tickfont=dict(size=14))
                fig1.update_yaxes(tickfont=dict(size=14))
                st.plotly_chart(fig1)

            with col23:
                st.write('<div style="text-align: center;"><h2>Total Available Apt.<br>per District</h2></div>', unsafe_allow_html=True)
                st.write(" ")
                x2 = filtered2.groupby(["area"]).size().reset_index(name="count")["count"].sort_values(ascending = 1).reset_index(drop=1)
                y2 = filtered2.groupby(["area"]).size().reset_index()["area"]
                fig2 = go.Figure([go.Bar(
                                        x = x2,
                                        y = y2,
                                        orientation='h',
                width = 0.5,
                text=x2.round(0),
                textposition='outside',
                marker=dict(color=x2, colorscale='viridis', line=dict(width=1.25, color='white')),
                textfont=dict(size=14))])
                fig2.update_layout(
                autosize=True,  # Automatically adjust size to fit entire layout
                margin=dict(l=0, r=2, t=0, b=0),
                xaxis=dict(range=[0, 1.9*max(x2)]))
                fig2.update_xaxes(tickfont=dict(size=14))
                fig2.update_yaxes(tickfont=dict(size=14))
                st.plotly_chart(fig2)
        with tab3:
            data_map(df1)
        st.write("_________")
        pass
    elif (choice == "Show Rent Prediction"):
        prediction()
dashboard()