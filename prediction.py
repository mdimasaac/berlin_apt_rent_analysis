def preprocessing(X,y):
    import numpy as np
    import pandas as pd
    df_corr = pd.concat([X,y], axis = 1).corr()
    cols_to_drop = df_corr[df_corr["rent"] < 0.8].index.tolist()
    while len(cols_to_drop) != 0:
        X = X.drop(columns = cols_to_drop)
        df_corr = pd.concat([X,y], axis = 1).corr()
        cols_to_drop = df_corr[df_corr["rent"] < 0.8].index.tolist()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 43)
    num_train = X_train.select_dtypes(np.number)
    num_test = X_test.select_dtypes(np.number)
    cat_train = X_train.select_dtypes(object)
    cat_test = X_test.select_dtypes(object)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler().fit(num_train)
    cols = scaler.get_feature_names_out(input_features = num_train.columns)
    num_train_scaled = scaler.transform(num_train)
    num_test_scaled = scaler.transform(num_test)
    num_train_scaled = pd.DataFrame(num_train_scaled, columns = cols)
    num_test_scaled = pd.DataFrame(num_test_scaled, columns = cols)

    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(handle_unknown = "ignore").fit(cat_train)
    cat_train = encoder.transform(cat_train).toarray()
    cat_test = encoder.transform(cat_test).toarray()
    cat_train_encoded = pd.DataFrame(cat_train)
    cat_test_encoded = pd.DataFrame(cat_test)
    cat_train_encoded = cat_train_encoded.reset_index(drop=True)
    cat_test_encoded = cat_test_encoded.reset_index(drop=True)
    X_train_processed = pd.concat([num_train_scaled, cat_train_encoded], axis=1)
    X_test_processed = pd.concat([num_test_scaled, cat_test_encoded], axis=1)
    return X_train_processed, X_test_processed, y_train, y_test

def xy_split():
    import numpy as np
    import pandas as pd
    df = pd.read_csv("data full clean.csv")
    df01 = df.iloc[:,:9]
    df02 = df.iloc[:,9:-1]
    df03 = df[["year"]]
    df01 = df01.drop(columns = ["link","addresse"])
    df02num = df02.select_dtypes(np.number)
    df02obj = df02.select_dtypes(object)

    for col in df02obj.columns:
        l = []
        for i in df02obj[col]:
            if i == "0":
                l.append("no info")
            else:
                l.append(i)
        df02obj[col] = l
    data = pd.concat([df01,df02num,df02obj,df03], axis = 1)
    col = ["pets_allowed","person_elevator"]
    for i in col:
        data[i] = data[i].astype("int64")
    X = data.drop(columns = ["rent"])
    y = data["rent"]
    return X,y

def prediction():
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from warnings import simplefilter
    from sklearn.exceptions import ConvergenceWarning
    simplefilter("ignore", category=ConvergenceWarning)

    X,y = xy_split()
    X_train_processed, X_test_processed, y_train, y_test = preprocessing(X,y)
    X_train_processed.columns = X_train_processed.columns.astype(str)
    X_test_processed.columns = X_test_processed.columns.astype(str)
    st.write('''<div style="text-align: center; color: yellow;"><h5>Predicting Rent Costs
     Using Different Machine Learning Models.</h5></div>''', unsafe_allow_html=True)
    st.write(" ")
    st.write(" ")
    import plotly.graph_objs as go
    x = list(range(y_test.shape[0]))
    y1 = y_test
    trace1 = go.Scatter(x=x,y=y1, mode = "lines", name = "actual price")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <style>
                .radio > label[for="radio"] {
                    display: inline-block;
                    margin-bottom: 0;
                    vertical-align: middle;
                }
                .radio input[type="radio"] {
                    display: none;
                }
                .radio input[type="radio"] + label:before {
                    content: "";
                    display: inline-block;
                    vertical-align: middle;
                    width: 20px;
                    height: 20px;
                    margin-right: 5px;
                    background-color: #fff;
                    border: 1px solid #888;
                    border-radius: 50%;
                }
                .radio input[type="radio"]:checked + label:before {
                    background-color: #888;
                }
            </style>
            """, unsafe_allow_html=True)
        options_left = ["Linear Regression","MLP","K-Neighbors","Decision Tree","Random Forest"]
        ml_left = st.radio(label='Choose Machine Learning Model:', options=options_left,
                            key = 'ml_left'
                            )
        st.write("_________")

        if ml_left == "Linear Regression":
            model_left = LinearRegression()
            st.subheader("Play Around with the Parameters!")
            col11,col12 = st.columns(2)
            with col11:
                fint_left = st.radio("fit_intercept:",("True","False"), key = "fint_left")
                if fint_left == "True":
                    fint_left = True
                else:
                    fint_left = False
            with col12:
                pos_left = st.radio("positive:",("True","False"), key = "pos_left")
                if pos_left == "True":
                    pos_left = True
                else:
                    pos_left = False
            st.subheader("Or Optimize Parameter")
            opt_left = st.checkbox("Optimize", key = "opt1_left")
            if opt_left:
                lfint_left = [True, False]
                lpos_left = [True, False]
                grid_left = {"fit_intercept":lfint_left,"positive":lpos_left}
                grid_search_left = GridSearchCV(estimator = model_left, param_grid = grid_left, cv = 5)
                grid_search_left.fit(X_test_processed, y_test)
                fint_left = grid_search_left.best_params_["fit_intercept"]
                pos_left = grid_search_left.best_params_["positive"]
            regr_left = LinearRegression(fit_intercept=fint_left, positive=pos_left)

        elif ml_left == "MLP":
            model_left = MLPRegressor()
            st.subheader("Play Around with the Parameters!")
            hlsize_left = st.slider("hidden_layer_sizes:", 50,300,50,50, key = "hlsize_left")
            maxiter_left = st.slider("max_iter:", 200,700,200,100, key = "maxiter_left")
            col11,col12 = st.columns(2)
            with col11:
                solver_left = st.radio("solver:", ("lbfgs","sgd","adam"), key = "solver_left")
            with col12:
                rate_left = st.radio("learning_rate:",("constant","invscaling","adaptive"), key = "rate_left")
            st.subheader("Or Optimize Parameter")
            opt_left = st.checkbox("Optimize", key = "opt2_left")
            if opt_left:
                lhlsize_left = list(range(200,301,100))
                lmaxiter_left = list(range(500,701,100))
                lsolver_left = ["lbfgs","sgd","adam"]
                lrate_left = ["constant","invscaling","adaptive"]
                grid_left = {"hidden_layer_sizes":lhlsize_left, "max_iter":lmaxiter_left,
                            "solver":lsolver_left, "learning_rate":lrate_left}
                grid_search_left = GridSearchCV(estimator = model_left, param_grid = grid_left, cv = 5)
                grid_search_left.fit(X_test_processed, y_test)
                hlsize_left = grid_search_left.best_params_["hidden_layer_sizes"]
                maxiter_left = grid_search_left.best_params_["max_iter"]
                solver_left = grid_search_left.best_params_["solver"]
                rate_left = grid_search_left.best_params_["learning_rate"]
            regr_left = MLPRegressor(hidden_layer_sizes = hlsize_left, max_iter = maxiter_left,
                                    solver = solver_left, learning_rate = rate_left)
    
        elif ml_left == "K-Neighbors":
            model_left = KNeighborsRegressor()
            st.subheader("Play Around with the Parameters!")
            n_left = st.slider("n_neighbors:", 3,9,3,1, key = "n_left")
            weight_left = st.radio("weight:", ("uniform","distance"), key = "weight_left")
            alg_left = "auto"
            st.subheader("Or Optimize Parameter")
            opt_left = st.checkbox("Optimize", key = "opt3_left")
            if opt_left:
                ln_left = list(range(3,10,1))
                lweight_left = ["uniform","distance"]
                lalg_left = ["auto"]
                grid_left = {"n_neighbors":ln_left, "weights":lweight_left, "algorithm":lalg_left}
                grid_search_left = GridSearchCV(estimator = model_left, param_grid = grid_left, cv = 5)
                grid_search_left.fit(X_test_processed, y_test)
                n_left = grid_search_left.best_params_["n_neighbors"]
                weight_left = grid_search_left.best_params_["weights"]
                alg_left = "auto"
            regr_left = KNeighborsRegressor(n_neighbors = n_left, weights = weight_left, algorithm = alg_left)
        
        elif ml_left == "Decision Tree":
            model_left = DecisionTreeRegressor()
            st.subheader("Play Around with the Parameters!")
            minsplit_left = st.slider("min_samples_split:",2,10,2,1, key = "split1_left")
            minleaf_left = st.slider("min_samples_leaf:", 2,10,2,1, key = "leaf1_left")
            col11,col12,col13 = st.columns([1,1,2])
            with col11:
                depth_left = st.radio("max_depth:", (3,4,5,6,7,None), key = "depth1_left")
            with col12:
                feat_left = st.radio("max_features:", (2,3,4,None), key = "feat1_left")
            with col13:
                crit_left = st.radio("criterion:",('squared_error','absolute_error'), key = "crit1_left")
            st.subheader("Or Optimize Parameter")
            opt_left = st.checkbox("Optimize", key = "opt4_left")
            if opt_left:
                lminsplit_left = list(range(4,8))
                lminleaf_left = list(range(4,8))
                ldepth_left = [6,7,None]
                lfeat_left = [3,4,None]
                lcrit_left = ['squared_error','absolute_error']
                grid_left = {"min_samples_split":lminsplit_left,"min_samples_leaf":lminleaf_left,
                            "max_depth":ldepth_left,"max_features":lfeat_left,"criterion":lcrit_left}
                with st.spinner("Running Algorithm.."):
                    grid_search_left = GridSearchCV(estimator = model_left, param_grid = grid_left, cv = 5)
                grid_search_left.fit(X_test_processed, y_test)
                minsplit_left = grid_search_left.best_params_["min_samples_split"]
                minleaf_left = grid_search_left.best_params_["min_samples_leaf"]
                depth_left = grid_search_left.best_params_["max_depth"]
                feat_left = grid_search_left.best_params_["max_features"]
                crit_left = grid_search_left.best_params_["criterion"]
            regr_left = DecisionTreeRegressor(min_samples_split = minsplit_left,
                        min_samples_leaf = minleaf_left, max_depth = depth_left, 
                        max_features = feat_left, criterion = crit_left)
        
        elif ml_left == "Random Forest":
            model_left = RandomForestRegressor()
            st.subheader("Play Around with the Parameters!")
            minsplit_left = st.slider("min_samples_split:",2,10,2,1, key = "split2_left")
            minleaf_left = st.slider("min_samples_leaf:", 2,10,2,1, key = "leaf2_left")
            col11,col12,col13 = st.columns([1,1,2])
            with col11:
                depth_left = st.radio("max_depth:", (3,4,5,6,7,None), key = "depth2_left")
            with col12:
                feat_left = st.radio("max_features:", (2,3,4,None), key = "feat2_left")
            with col13:
                crit_left = st.radio("criterion:",('squared_error','absolute_error'), key = "crit2_left")
            st.subheader("Or Optimize Parameter")
            opt_left = st.checkbox("Optimize", key = "opt5_left")
            if opt_left:
                lminsplit_left = list(range(4,8))
                lminleaf_left = list(range(4,8))
                ldepth_left = [6,7,None]
                lfeat_left = [3,4,None]
                lcrit_left = ['squared_error','absolute_error']
                grid_left = {"min_samples_split":lminsplit_left,"min_samples_leaf":lminleaf_left,
                            "max_depth":ldepth_left,"max_features":lfeat_left,"criterion":lcrit_left}
                with st.spinner("Running Algorithm.."):
                    grid_search_left = GridSearchCV(estimator = model_left, param_grid = grid_left, cv = 5)
                grid_search_left.fit(X_test_processed, y_test)
                minsplit_left = grid_search_left.best_params_["min_samples_split"]
                minleaf_left = grid_search_left.best_params_["min_samples_leaf"]
                depth_left = grid_search_left.best_params_["max_depth"]
                feat_left = grid_search_left.best_params_["max_features"]
                crit_left = grid_search_left.best_params_["criterion"]
            regr_left = RandomForestRegressor(min_samples_split = minsplit_left,
                        min_samples_leaf = minleaf_left, max_depth = depth_left, 
                        max_features = feat_left, criterion = crit_left)

    with col2:
        st.markdown("""
            <style>
                .radio > label[for="radio"] {
                    display: inline-block;
                    margin-bottom: 0;
                    vertical-align: middle;
                }
                .radio input[type="radio"] {
                    display: none;
                }
                .radio input[type="radio"] + label:before {
                    content: "";
                    display: inline-block;
                    vertical-align: middle;
                    width: 20px;
                    height: 20px;
                    margin-right: 5px;
                    background-color: #fff;
                    border: 1px solid #888;
                    border-radius: 50%;
                }
                .radio input[type="radio"]:checked + label:before {
                    background-color: #888;
                }
            </style>
            """, unsafe_allow_html=True)
        options_right = ["Linear Regression","MLP","K-Neighbors","Decision Tree","Random Forest"]
        ml_right = st.radio(label='Choose Machine Learning Model:', options=options_right, key='ml_right')
        st.write("_________")

        if ml_right == "Linear Regression":
            model_right = LinearRegression()
            st.subheader("Play Around with the Parameters!")
            col21,col22 = st.columns(2)
            with col21:
                fint_right = st.radio("fit_intercept:",("True","False"), key = "fint_right")
                if fint_right == "True":
                    fint_right = True
                else:
                    fint_right = False
            with col22:
                pos_right = st.radio("positive:",("True","False"), key = "pos_right")
                if pos_right == "True":
                    pos_right = True
                else:
                    pos_right = False
            st.subheader("Or Optimize Parameter")
            opt_right = st.checkbox("Optimize", key = "opt1_right")
            if opt_right:
                lfint_right = [True, False]
                lpos_right = [True, False]
                grid_right = {"fit_intercept":lfint_right,"positive":lpos_right}
                grid_search_right = GridSearchCV(estimator = model_right, param_grid = grid_right, cv = 5)
                grid_search_right.fit(X_test_processed, y_test)
                fint_right = grid_search_right.best_params_["fit_intercept"]
                pos_right = grid_search_right.best_params_["positive"]
            regr_right = LinearRegression(fit_intercept=fint_right, positive=pos_right)

        elif ml_right == "MLP":
            model_right = MLPRegressor()
            st.subheader("Play Around with the Parameters!")
            hlsize_right = st.slider("hidden_layer_sizes:", 50,300,50,50, key = "hlsize_right")
            maxiter_right = st.slider("max_iter:", 200,700,200,100, key = "maxiter_right")
            col21,col22 = st.columns(2)
            with col21:
                solver_right = st.radio("solver:", ("lbfgs","sgd","adam"), key = "solver_right")
            with col22:
                rate_right = st.radio("learning_rate:",("constant","invscaling","adaptive"), key = "rate_right")
            st.subheader("Or Optimize Parameter")
            opt_right = st.checkbox("Optimize", key = "opt2_right")
            if opt_right:
                lhlsize_right = list(range(200,301,100))
                lmaxiter_right = list(range(500,701,100))
                lsolver_right = ["lbfgs","sgd","adam"]
                lrate_right = ["constant","invscaling","adaptive"]
                grid_right = {"hidden_layer_sizes":lhlsize_right, "max_iter":lmaxiter_right,
                            "solver":lsolver_right, "learning_rate":lrate_right}
                grid_search_right = GridSearchCV(estimator = model_right, param_grid = grid_right, cv = 5)
                grid_search_right.fit(X_test_processed, y_test)
                hlsize_right = grid_search_right.best_params_["hidden_layer_sizes"]
                maxiter_right = grid_search_right.best_params_["max_iter"]
                solver_right = grid_search_right.best_params_["solver"]
                rate_right = grid_search_right.best_params_["learning_rate"]
            regr_right = MLPRegressor(hidden_layer_sizes = hlsize_right, max_iter = maxiter_right,
                                    solver = solver_right, learning_rate = rate_right)
        
        elif ml_right == "K-Neighbors":
            model_right = KNeighborsRegressor()
            st.subheader("Play Around with the Parameters!")
            n_right = st.slider("n_neighbors:", 3,9,3,1, key = "n_right")
            weight_right = st.radio("weight:", ("uniform","distance"), key = "weight_right")
            alg_right = "auto"
            st.subheader("Or Optimize Parameter")
            opt_right = st.checkbox("Optimize", key = "opt3_right")
            if opt_right:
                ln_right = list(range(3,10,1))
                lweight_right = ["uniform","distance"]
                lalg_right = ["auto"]
                grid_right = {"n_neighbors":ln_right, "weights":lweight_right, "algorithm":lalg_right}
                grid_search_right = GridSearchCV(estimator = model_right, param_grid = grid_right, cv = 5)
                grid_search_right.fit(X_test_processed, y_test)
                n_right = grid_search_right.best_params_["n_neighbors"]
                weight_right = grid_search_right.best_params_["weights"]
                alg_right = "auto"
            regr_right = KNeighborsRegressor(n_neighbors = n_right, weights = weight_right, algorithm = alg_right)
        
        elif ml_right == "Decision Tree":
            model_right = DecisionTreeRegressor()
            st.subheader("Play Around with the Parameters!")
            minsplit_right = st.slider("min_samples_split:",2,10,2,1, key = "split1_right")
            minleaf_right = st.slider("min_samples_leaf:", 2,10,2,1, key = "leaf1_right")
            col21,col22,col23 = st.columns([1,1,2])
            with col21:
                depth_right = st.radio("max_depth:", (3,4,5,6,7,None), key = "depth1_right")
            with col22:
                feat_right = st.radio("max_features:", (2,3,4,None), key = "feat1_right")
            with col23:
                crit_right = st.radio("criterion:",('squared_error','absolute_error'), key = "crit1_right")
            st.subheader("Or Optimize Parameter")
            opt_right = st.checkbox("Optimize", key = "opt4_right")
            if opt_right:
                lminsplit_right = list(range(4,8))
                lminleaf_right = list(range(4,8))
                ldepth_right = [6,7,None]
                lfeat_right = [3,4,None]
                lcrit_right = ['squared_error','absolute_error']
                grid_right = {"min_samples_split":lminsplit_right,"min_samples_leaf":lminleaf_right,
                            "max_depth":ldepth_right,"max_features":lfeat_right,"criterion":lcrit_right}
                with st.spinner("Running Algorithm.."):
                    grid_search_right = GridSearchCV(estimator = model_right, param_grid = grid_right, cv = 5)
                grid_search_right.fit(X_test_processed, y_test)
                minsplit_right = grid_search_right.best_params_["min_samples_split"]
                minleaf_right = grid_search_right.best_params_["min_samples_leaf"]
                depth_right = grid_search_right.best_params_["max_depth"]
                feat_right = grid_search_right.best_params_["max_features"]
                crit_right = grid_search_right.best_params_["criterion"]
            regr_right = DecisionTreeRegressor(min_samples_split = minsplit_right,
                        min_samples_leaf = minleaf_right, max_depth = depth_right, 
                        max_features = feat_right, criterion = crit_right)
        
        elif ml_right == "Random Forest":
            model_right = RandomForestRegressor()
            st.subheader("Play Around with the Parameters!")
            minsplit_right = st.slider("min_samples_split:",2,10,2,1, key = "split2_right")
            minleaf_right = st.slider("min_samples_leaf:", 2,10,2,1, key = "leaf2_right")
            col21,col22,col23 = st.columns([1,1,2])
            with col21:
                depth_right = st.radio("max_depth:", (3,4,5,6,7,None), key = "depth2_right")
            with col22:
                feat_right = st.radio("max_features:", (2,3,4,None), key = "feat2_right")
            with col23:
                crit_right = st.radio("criterion:",('squared_error','absolute_error'), key = "crit2_right")
            st.subheader("Or Optimize Parameter")
            opt_right = st.checkbox("Optimize", key = "opt5_right")
            if opt_right:
                lminsplit_right = list(range(4,8))
                lminleaf_right = list(range(4,8))
                ldepth_right = [6,7,None]
                lfeat_right = [3,4,None]
                lcrit_right = ['squared_error','absolute_error']
                grid_right = {"min_samples_split":lminsplit_right,"min_samples_leaf":lminleaf_right,
                            "max_depth":ldepth_right,"max_features":lfeat_right,"criterion":lcrit_right}
                with st.spinner("Running Algorithm.."):
                    grid_search_right = GridSearchCV(estimator = model_right, param_grid = grid_right, cv = 5)
                grid_search_right.fit(X_test_processed, y_test)
                minsplit_right = grid_search_right.best_params_["min_samples_split"]
                minleaf_right = grid_search_right.best_params_["min_samples_leaf"]
                depth_right = grid_search_right.best_params_["max_depth"]
                feat_right = grid_search_right.best_params_["max_features"]
                crit_right = grid_search_right.best_params_["criterion"]
            regr_right = RandomForestRegressor(min_samples_split = minsplit_right,
                        min_samples_leaf = minleaf_right, max_depth = depth_right, 
                        max_features = feat_right, criterion = crit_right)
    st.write("_________")

    col01,col02 = st.columns(2)
    with col01:
        with st.spinner('Building Machine Learning Model...'):
            st.write('<div style="text-align: center; color: yellow;"><h3>'+ml_left+'</h3></div>', unsafe_allow_html=True)
            regr_left.fit(X_train_processed, y_train)
            pred_test_left = regr_left.predict(X_test_processed)
            mean_score_train = np.mean(cross_val_score(regr_left, X_train_processed, y_train, cv = 5))
            mean_score_test = np.mean(cross_val_score(regr_left, X_test_processed, y_test, cv = 5))
            layout_left = go.Layout()
            y2_left = pred_test_left
            trace2_left = go.Scatter(x=x,y=y2_left, mode = "lines", name = "predicted price")
            data_left = [trace1, trace2_left]
            fig_left = go.Figure(data=data_left, layout = layout_left)
            fig_left.update_layout(template="plotly_dark", autosize=True, yaxis_title = "Rent in €", xaxis_title = "Apartment-ID")
            fig_left.update_layout(legend=dict(
                orientation="h",
                entrywidth=70,
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))
            st.plotly_chart(fig_left)
            mean_score_test_left = np.mean(cross_val_score(regr_left, X_test_processed, y_test, cv = 5))
            st.write('<div style="text-align: left; color: yellow;"><h4>Prediction Score: '+str(round(mean_score_test_left,3))+'</h4></div>', unsafe_allow_html=True)

    with col02:
        with st.spinner('Building Machine Learning Model...'):
            st.write('<div style="text-align: center; color: yellow;"><h3>'+ml_right+'</h3></div>', unsafe_allow_html=True)
            regr_right.fit(X_train_processed, y_train)
            pred_test_right = regr_right.predict(X_test_processed)
            mean_score_train = np.mean(cross_val_score(regr_right, X_train_processed, y_train, cv = 5))
            mean_score_test = np.mean(cross_val_score(regr_right, X_test_processed, y_test, cv = 5))
            layout_right = go.Layout()
            y2_right = pred_test_right
            trace2_right = go.Scatter(x=x,y=y2_right, mode = "lines", name = "predicted price")
            data_right = [trace1, trace2_right]
            fig_right = go.Figure(data=data_right, layout = layout_right)
            fig_right.update_layout(template="plotly_dark", autosize=True)
            fig_right.update_layout(legend=dict(
                orientation="h",
                entrywidth=70,
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))
            st.plotly_chart(fig_right)
            mean_score_test_right = np.mean(cross_val_score(regr_right, X_test_processed, y_test, cv = 5))
            st.write('<div style="text-align: left; color: yellow;"><h4>Prediction Score: '+str(round(mean_score_test_right,3))+'</h4></div>', unsafe_allow_html=True)