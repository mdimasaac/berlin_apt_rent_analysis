def plot_map(df1, col):
    # Create the choropleth map
    import plotly.graph_objs as go
    import streamlit as st
    from streamlit_plotly_events import plotly_events

    if col == "Total Available Apartments":
        text = "Available Apartments:"
        col = "available_apartments"
    elif col == "Average Monthly Rent":
        text = "Average Monthly Rent:"
        col = "mean_rent"
    elif col == "Highest Monthly Rent":
        text = "Highest Monthly Rent:"
        col = "max_rent"
    elif col == "Lowest Monthly Rent":
        text = "Lowest Monthly Rent:"
        col = "min_rent"
    elif col == "Most Frequent Monthly Rent":
        text = "Most Frequent Monthly Rent:"
        col = "mode_rent"
    elif col == "Newest Building":
        text = "Newest Building:"
        col = "newest_building"
    elif col == "Oldest Building":
        text = "Oldest Building:"
        col = "oldest_building"
    fig = go.Figure(go.Choroplethmapbox(geojson=df1.__geo_interface__,
                                        locations=df1.index,
                                        z=df1[col].tolist(),
                                        colorscale="Viridis",
                                        # color_continuous_scale="Viridis",
                                        zmin=df1[col].min(),
                                        zmax=df1[col].max(),
                                        marker_opacity=0.4,
                                        marker_line_width=1,
                                        name = "",
                                        text=df1[col],
                                        hovertemplate = "<b>%{location}</b><br>" + text + 
                                                        " %{z}<br>" + "<extra></extra>"
                                        ))
    fig.update_layout(mapbox_style='carto-positron',
                    mapbox_zoom=8.8,
                    mapbox_center={'lat': 52.5167, 'lon': 13.3833})
    fig.update_layout(margin={'r':0,'t':0,'l':0,'b':0},template="plotly_dark" )
    fig.update_traces(hoverinfo='text')  # disable hoverinfo
    fig.update_layout(showlegend=False)
    st.write("HINT: Click on the District on the map to show detailed information for each District.")
    selected_points = plotly_events(fig)
    if len(selected_points) != 0:
        ind = list(selected_points[0].keys())[2]
        index = selected_points[0][ind]
    else:
        ind = 0
        index = 0
    return index
    
def data_map(df1):
    import streamlit as st
    options = ["Total Available Apartments","Average Monthly Rent","Highest Monthly Rent","Lowest Monthly Rent",
                "Most Frequent Monthly Rent","Newest Building","Oldest Building"]
    option = st.selectbox("Select an option", options)
    min_rent=max_rent=mean_rent=mode_rent=available_apt="-"
    index = plot_map(df1,option)
    with st.expander("Show Details"):
        district_name = df1.index.values[index]
        st.write('<div style="text-align: center;"><h3>District: '+district_name+'</h3></div>', unsafe_allow_html=True)
        st.write("_________")
        col1,col2,col3,col4,col5 = st.columns(5)
        with col1:
            available_apt = df1.iloc[index]["available_apartments"]
            st.write('<div style="text-align: center;"><h5>Available apt.</h5></div>', unsafe_allow_html=True)
            st.write('<div style="text-align: center;"><h1>'+str(available_apt)+'</h1></div>', unsafe_allow_html=True)
        with col2:
            min_rent = df1.iloc[index]["min_rent"]
            st.write('<div style="text-align: center;"><h5>Lowest rent</h5></div>', unsafe_allow_html=True)
            st.write('<div style="text-align: center;"><h1>'+str(int(min_rent))+' €</h1></div>', unsafe_allow_html=True)
        with col3:
            max_rent = df1.iloc[index]["max_rent"]
            st.write('<div style="text-align: center;"><h5>Highest rent</h5></div>', unsafe_allow_html=True)
            st.write('<div style="text-align: center;"><h1>'+str(int(max_rent))+' €</h1></div>', unsafe_allow_html=True)
        with col4:
            mode_rent = df1.iloc[index]["mode_rent"]
            st.write('<div style="text-align: center;"><h5>Popular rent</h5></div>', unsafe_allow_html=True)
            st.write('<div style="text-align: center;"><h1>'+str(int(mode_rent))+' €</h1></div>', unsafe_allow_html=True)
        with col5:
            mean_rent = df1.iloc[index]["mean_rent"]
            st.write('<div style="text-align: center;"><h5>Average rent</h5></div>', unsafe_allow_html=True)
            st.write('<div style="text-align: center;"><h1>'+str(int(mean_rent))+' €</h1></div>', unsafe_allow_html=True)
    

