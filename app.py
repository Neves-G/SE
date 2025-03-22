import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

import plotly.express as px
import pandas as pd
import joblib
import numpy as np
from sklearn import metrics

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

# Load stylesheets, data and model
df_17_18 = pd.read_csv('dash-app/data/IST_Civil_Pav_2017_2018.csv')
df_19 = pd.read_csv('dash-app/data/2019Test.csv')
features_df = pd.read_csv('dash-app/data/Features.csv')
features_19_df = pd.read_csv('dash-app/data/Features_19.csv')
ModelNN = joblib.load('dash-app/models/NN_model.pkl')

# Rename columns
df_17_18.rename(columns={'Date_start': 'Date','Power_kW':'Power (kW)','Hour': 'Hour',
                         'Power-1': 'Power -1h (kW)','Rolling3': '3h Rolling Average',
                         'temp_C': 'Temperature (ºC)','solarRad_W/m2': 'Radiance (W/m^2)',
                         'Convolution': 'Convolution Feature'}, inplace=True)
df_19.rename(columns={'Date': 'Date','Civil (kWh)':'Power (kW)','Hour': 'Hour',
                         'Power-1': 'Power -1h (kW)','Rolling3': '3h Rolling Average',
                         'temp_C': 'Temperature (ºC)','solarRad_W/m2': 'Radiance (W/m^2)',
                         'Convolution': 'Convolution Feature'}, inplace=True)
features_df.rename(columns={'Date_start': 'Date','Power_kW':'Power (kW)','Hour': 'Hour',
                            'Power-1': 'Power -1h (kW)','Rolling3': '3h Rolling Average',
                            'temp_C': 'Temperature (ºC)','solarRad_W/m2': 'Radiance (W/m^2)',
                            'Convolution': 'Convolution Feature', 'CyclicHour':'Cyclic Hour',
                            'DayOfWeek': 'Day of the Week', 'WorkDay': 'Workday', 'LunchTime': 'Lunch Time',
                            'Holidays':'Holidays','Classes':'Class Time'}, inplace=True)
features_19_df.rename(columns={'Date_start': 'Date','Power_kW':'Power (kW)','Hour': 'Hour',
                            'Power-1': 'Power -1h (kW)','Rolling3': '3h Rolling Average',
                            'temp_C': 'Temperature (ºC)','solarRad_W/m2': 'Radiance (W/m^2)',
                            'Convolution': 'Convolution Feature', 'CyclicHour':'Cyclic Hour',
                            'DayOfWeek': 'Day of the Week', 'WorkDay': 'Workday', 'LunchTime': 'Lunch Time',
                            'Holidays':'Holidays','Classes':'Class Time'}, inplace=True)

# Set date to index
df_17_18.set_index('Date', inplace=True)
df_19.set_index('Date', inplace=True)
features_df.set_index('Date', inplace=True)
features_19_df.set_index('Date', inplace=True)

# Make normalized dfs to plot in the same axis
df_17_18_norm = (df_17_18 - df_17_18.mean()) / df_17_18.std()
df_19_norm = (df_19 - df_19.mean()) / df_19.std()

# Use model on 2019 data
X_test = df_19.drop('Power (kW)', axis=1)
Y_test = df_19['Power (kW)']
PredictionsNN = ModelNN.predict(X_test.values)

# Join predictions to 2019 data
df_19['Predictions (kW)'] = PredictionsNN

# Make copy of 2019 data for the user to play with
df_19_train = df_19.copy()

def generate_table(dataframe, max_rows=10, style1={'width': '100%', 'tableLayout': 'auto'}, style2={'overflowX': 'auto', 'display': 'block', 'maxWidth': '100%'}):
    return html.Div([
        html.Div([
            html.Table([
                html.Thead(
                    html.Tr([html.Th(dataframe.index.name)] +
                            [html.Th(col) for col in dataframe.columns])
                ),
                html.Tbody([
                    html.Tr([html.Td(dataframe.index[i])] +
                            [html.Td(dataframe.iloc[i][col]) for col in dataframe.columns])
                    for i in range(min(len(dataframe), max_rows))
                ])
            ], style=style1)  # Allow flexible column widths
        ], style=style2)  # Horizontal scroll for the table only
    ])

def generate_graph(dataframe, y_col, normalize=False):
    dataframe = dataframe.drop('Hour', axis=1)
    if y_col == 'all':
        grph = px.line(dataframe)
    else:
        if isinstance(y_col, list):
            grph = px.line(dataframe, y=y_col)
        else:
            grph = px.line(dataframe, y=[y_col])
    grph.update_xaxes(title_text='Date')
    if normalize:
        grph.update_yaxes(title_text='Normalized Data')
    else:
        grph.update_yaxes(title_text='Power (kWh)')
    grph.update_layout(showlegend=True)

    return grph

def metrics_df(test, predictions, model='Neural Network'):
    mtrsc = {model: [], 'Reference': [None, None, None, 20, 0.2, 0.05]}
    
    MAE = metrics.mean_absolute_error(test,predictions) 
    MBE = np.mean(test-predictions)
    MSE = metrics.mean_squared_error(test,predictions)  
    RMSE = np.sqrt(metrics.mean_squared_error(test,predictions))
    cvRMSE = RMSE/np.mean(test)
    NMBE = MBE/np.mean(test)

    mtrsc[model] = [MAE, MBE, MSE, RMSE, cvRMSE, NMBE]
    metrics_df = pd.DataFrame(mtrsc, index=['MAE', 'MBE', 'MSE', 'RMSE', 'cvRMSE', 'NMBE'])
    return metrics_df

app = dash.Dash(__name__, assets_folder='assets', suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Store(id='n-clicks-store', data=0),
    dcc.Store(id='n-clicks-store-2', data=0),
    dcc.Tabs(id='tabs', value='tab-home', className='top-tabs', children=[
        dcc.Tab(label='Home', value='tab-home'),
        dcc.Tab(label='Training Data', value='tab-1'),
        dcc.Tab(label='Forecast Data', value='tab-2'),
        dcc.Tab(label='Forecast Model', value='tab-3'),
        dcc.Tab(label='Forecast Tools', value='tab-4'),
        dcc.Tab(label='Features Lab', value='tab-5')
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-home':
        return html.Div([
            html.Div([
                html.P(''),
                html.H1('Welcome to the IST Civil Building Energy Consumption Dashboard!', style={'marginBottom':'20px'}),
                html.P('This dashboard provides insights into the energy consumption of the IST Civil Building. '
                       'You can explore the training data, forecast data, and model predictions. '
                       'Use the tabs above to navigate through different sections of the dashboard.'),
                html.P('In the "Training Data" tab, you can view and analyze the historical energy consumption data for 2017 and 2018.'),
                html.P('In the "Forecast Data" tab, you can view the forecasted energy consumption data for 2019.'),
                html.P('In the "Forecast Model" tab, you can view the predictions made by the model.'),
                html.P('In the "Forecast Tools" tab, you can select features and train models to predict energy consumption.'),
                html.P('In the "Features Lab" tab, you can engineer your own features to train a model.',
                       style={'marginBottom': '20px'}),
                html.P([
                    "This dashboard was created by ",
                    html.A("Guilherme Neves", href="https://fenix.tecnico.ulisboa.pt/homepage/ist1102548", target="_blank", style={'color': '#FFB703', 'textDecoration': 'none'}),
                    "."
                ])
            ], style={'width': '60%', 'display': 'inline-block', 'justifyContent': 'center'}),
            html.Div([
                html.Img(src='/assets/IST_A_RGB_POS.jpg', style={'height': '400px'})
            ], style={'width': '40%', 'display': 'inline-block', 'textAlign': 'right'})
        ], style={'display': 'flex', 'justifyContent': 'center'})
    
    elif tab == 'tab-1':
        return html.Div([
            html.H3('Visualize the 2017/18 energy consumption data, and the features we have chosen to train our model on.'),
            dcc.Tabs(
                id='17_18-data',
                value='graph',
                children=[
                    dcc.Tab(label='Graph', value='graph'),
                    dcc.Tab(label='Table', value='table')
                ]
            ),
            html.Div(id='17_18-data-output')
        ])

    elif tab == 'tab-2':
        return html.Div([
            html.H3('Visualize the 2019 energy consumption data, and the features we have chosen to train our model on.'),
            dcc.Tabs(
                id='19-data',
                value='graph',
                children=[
                    dcc.Tab(label='Graph', value='graph'),
                    dcc.Tab(label='Table', value='table')
                ]
            ),
            html.Div(id='19-data-output')
        ])
    
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Visualize the characteristics of our best predictor model for the 2019 power consumption in the Civil building.'),
            dcc.Tabs(
                id='forecast-data',
                value='graph',
                children=[
                    dcc.Tab(label='Graph', value='graph'),
                    dcc.Tab(label='Table', value='table'),
                    dcc.Tab(label='Metrics', value='metrics')
                ]
            ),
            html.H2('Neural Network Model', style={'textAlign': 'center'}),
            html.Div(id='forecast-data-output')
        ])
    
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Train your own model to predict energy consumption!'),
            html.H4('Select the features you want to use to train the model and the model type.'),
            dcc.Dropdown(
                id='features-tools',
                options=[
                    {'label': 'Temperature', 'value': 'Temperature (ºC)'},
                    {'label': 'Radiance', 'value': 'Radiance (W/m^2)'},
                    {'label': 'Convolution Feature', 'value': 'Convolution Feature'},
                    {'label': '3h Rolling Average', 'value': '3h Rolling Average'},
                    {'label': 'Power -1h', 'value': 'Power -1h (kW)'},
                    {'label': 'Hour', 'value': 'Hour'}
                    #{'label': 'Custom', 'value': 'Custom'}
                ],
                placeholder='Select features',
                multi=True
            ),
            dcc.RadioItems(
                id='model',
                options=[
                    {'label': 'Neural Network', 'value': 'NN'},
                    {'label': 'Random Forest', 'value': 'RF'}
                ],
                value='NN'
            ),
            html.Div(id='train'),
            html.Div(id='output-predict')
        ])
    
    elif tab == 'tab-5':
       return html.Div([
            html.H3('Engineer your own features to train a model!'),
            html.H5('Use the available features to engineer your own!'),
            html.P('Select features to multiply and specify a coefficient for each multiplication.'),
            html.Div([
                html.Label('Select features to multiply:'),
                dcc.Dropdown(
                    id='multiply-features-dropdown',
                    options=[
                        {'label': 'Temperature (ºC)', 'value': 'Temperature (ºC)'},
                        {'label': 'Radiance (W/m^2)', 'value': 'Radiance (W/m^2)'},
                        {'label': 'Convolution Feature', 'value': 'Convolution Feature'},
                        {'label': '3h Rolling Average', 'value': '3h Rolling Average'},
                        {'label': 'Power -1h', 'value': 'Power -1h (kW)'},
                        {'label': 'Hour', 'value': 'Hour'},
                        {'label': 'Cyclic Hour', 'value': 'Cyclic Hour'},
                        {'label': 'Day of the Week', 'value': 'Day of the Week'},
                        {'label': 'Workday', 'value': 'Workday'},
                        {'label': 'Lunch Time', 'value': 'Lunch Time'},
                        {'label': 'Holidays', 'value': 'Holidays'},
                        {'label': 'Class Time', 'value': 'Class Time'}
                    ],
                    multi=True,
                    placeholder='Select desired features'
                ),
                html.Label('Coefficient for the product:'),
                dcc.Input(id='product-coefficient', type='number', placeholder='Enter coefficient'),
                html.Button('Save Multiplication', id='save-multiplication', n_clicks=0, style={'marginTop': '10px', 'meginLeft': '10px', 'marginRight': '10px'}),
            ], style={'marginBottom': '20px'}),
            html.Div(id='saved-multiplications', style={'marginBottom': '20px'}),  # Display saved multiplications
            html.Button('Clear Multiplications', id='clear-multiplications', n_clicks=0, style={'marginBottom': '10px', 'display': 'block'}),
            html.Button('Add Feature', id='add-custom-feature', n_clicks=0, style={'marginTop': '10px', 'display': 'block', 'marginBottom': '15px'}),
            html.Div(id='custom-feature-output'),
            html.Div(id='train-custom-output')
        ])

# Tab 1 callbacks
@app.callback(
    Output('17_18-data-output', 'children'),
    Input('17_18-data', 'value')
)
def update_output(value):
    if value == 'table':
        return generate_table(df_17_18)
    elif value == 'graph':
        return dcc.Graph(figure=generate_graph(df_17_18_norm, 'all', normalize=True))

# Tab 2 callbacks
@app.callback(
    Output('19-data-output', 'children'),
    Input('19-data', 'value')
)
def update_output(value):
    if value == 'table':
        return generate_table(df_19.drop('Predictions (kW)', axis=1))
    elif value == 'graph':
        return dcc.Graph(figure=generate_graph(df_19_norm, 'all', normalize=True))

# Tab 3 callbacks
@app.callback(
    Output('forecast-data-output', 'children'),
    Input('forecast-data', 'value')
)
def update_output(value):
    if value == 'table':
        return generate_table(df_19.drop(['Hour', 'Power -1h (kW)', '3h Rolling Average',
                                          'Convolution Feature', 'Temperature (ºC)', 'Radiance (W/m^2)'], axis=1), style1={'margin': 'auto'},style2={})
    elif value == 'graph':
        return dcc.Graph(figure=generate_graph(df_19, ['Power (kW)','Predictions (kW)']))
    elif value == 'metrics':
        return generate_table(metrics_df(Y_test, PredictionsNN),style1={'margin': 'auto'},style2={})

# Tab 4 callbacks
@app.callback(
    Output('train', 'children'),
    Input('model', 'value'),
    Input('features-tools', 'value'),
    State('n-clicks-store', 'data')
)
def selection(model, features, stored_n_clicks):
    if not features or model is None:
        return html.P('Please select at least one feature and the model type to train.', style={'margin':'5px'})

    return html.Div([
        html.Button('Train Model', id='train-button', n_clicks=stored_n_clicks),
    ])

@app.callback(
    Output('output-predict', 'children'),
    Output('n-clicks-store', 'data'),
    Input('train-button', 'n_clicks'),
    State('n-clicks-store', 'data'),
    State('model', 'value'),
    State('features-tools', 'value')
)
def train_model(n_clicks, stored_n_clicks, model, features):
    # Ensure the button is pressed and n_clicks is greater than stored_n_clicks
    if n_clicks is None:
        n_clicks = 0  # Handle the case where n_clicks is None initially

    if n_clicks > stored_n_clicks:
        if not features or model is None:
            return html.P('Please select at least one feature and the model type to train.', style={'margin':'5px'}), n_clicks

        X_train = df_17_18[features]
        Y_train = df_17_18['Power (kW)']

        if model == 'RF':
            parameters = {'bootstrap': True,
                          'min_samples_leaf': 3,
                          'n_estimators': 200,
                          'min_samples_split': 15,
                          'max_features': 'sqrt',
                          'max_depth': 30,
                          'max_leaf_nodes': None}
            Model = RandomForestRegressor(**parameters)
            Model.fit(X_train.values, Y_train.values)
            Predictions = Model.predict(X_test[features].values)
            df_19_train['Predictions (kW)'] = Predictions
            return html.Div(
                [html.P('Random Forest model trained!'),
                 dcc.Tabs(
                    id='user-forecast-data',
                    value='graph',
                    children=[
                        dcc.Tab(label='Graph', value='graph'),
                        dcc.Tab(label='Table', value='table'),
                        dcc.Tab(label='Metrics', value='metrics')
                    ]
                ),
                    html.Div(id='user-forecast-output')]
            ), n_clicks

        if model == 'NN':
            Model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=100)
            Model.fit(X_train.values, Y_train.values)
            Predictions = Model.predict(X_test[features].values)
            df_19_train['Predictions (kW)'] = Predictions
            return html.Div(
                [html.P('Neural Network model trained!'),
                 dcc.Tabs(
                    id='user-forecast-data',
                    value='graph',
                    children=[
                        dcc.Tab(label='Graph', value='graph'),
                        dcc.Tab(label='Table', value='table'),
                        dcc.Tab(label='Metrics', value='metrics')
                    ]
                ),
                    html.Div(id='user-forecast-output')]
            ), n_clicks

    return dash.no_update, stored_n_clicks

@app.callback(
    Output('user-forecast-output', 'children'),
    Input('user-forecast-data', 'value'),
    State('model', 'value'),
)
def update_output(value, model):
    if value == 'table':
        return generate_table(df_19_train.drop(['Hour', 'Power -1h (kW)', '3h Rolling Average',
                                          'Convolution Feature', 'Temperature (ºC)', 'Radiance (W/m^2)'], axis=1),style1={'margin': 'auto'},style2={})
    elif value == 'graph':
        return dcc.Graph(figure=generate_graph(df_19_train, ['Power (kW)','Predictions (kW)']))
    elif value == 'metrics':
        Predictions = df_19_train['Predictions (kW)']
        return generate_table(metrics_df(Y_test, Predictions, model=model),style1={'margin': 'auto'},style2={})

# Tab 5 callbacks
@app.callback(
    Output('saved-multiplications', 'children'),
    Input('save-multiplication', 'n_clicks'),
    Input('clear-multiplications', 'n_clicks'),
    State('multiply-features-dropdown', 'value'),
    State('product-coefficient', 'value'),
    State('saved-multiplications', 'children')
)
def manage_multiplications(save_clicks, clear_clicks, selected_features, coefficient, saved_multiplications):
    ctx = dash.callback_context  # Get the context of the callback
    if not ctx.triggered:
        return dash.no_update

    # Determine which button triggered the callback
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'clear-multiplications':
        # Clear all saved multiplications
        return []

    if triggered_id == 'save-multiplication' and selected_features and coefficient is not None:
        # Save a new multiplication
        multiplication_description = f"{coefficient} * {' * '.join(selected_features)}"
        if saved_multiplications is None:
            saved_multiplications = []
        saved_multiplications.append(html.Div(multiplication_description))
        return saved_multiplications

    return dash.no_update

@app.callback(
    Output('custom-feature-output', 'children'),
    Input('add-custom-feature', 'n_clicks'),
    State('saved-multiplications', 'children')
)
def add_custom_features(n_clicks, saved_multiplications):
    if n_clicks > 0 and saved_multiplications:
        try:
            # Combine all multiplications into a single custom feature for features_df
            custom_feature = 0
            for multiplication in saved_multiplications:
                # Parse the multiplication description
                description = multiplication['props']['children']
                coefficient, *features = description.split(' * ')
                coefficient = float(coefficient)
                product = coefficient
                for feature in features:
                    product *= features_df[feature]
                custom_feature += product

            # Add the custom feature to features_df
            features_df['Custom'] = custom_feature

            # Combine all multiplications into a single custom feature for features_19_df
            custom_feature_19 = 0
            for multiplication in saved_multiplications:
                # Parse the multiplication description
                description = multiplication['props']['children']
                coefficient, *features = description.split(' * ')
                coefficient = float(coefficient)
                product_19 = coefficient
                for feature in features:
                    product_19 *= features_19_df[feature]
                custom_feature_19 += product_19

            # Add the custom feature to features_19_df
            features_19_df['Custom'] = custom_feature_19

            return html.Div([
                dcc.Tabs(
                    id='features-custom',
                    value='graph',
                    children=[
                        dcc.Tab(label='Graph', value='graph'),
                        dcc.Tab(label='Table', value='table')
                    ]
                ),
                html.Div(id='custom-features-output-2')
            ])
        except Exception as e:
            return html.P(f'Error: {str(e)}')
    return dash.no_update

@app.callback(
    Output('custom-features-output-2', 'children'),
    Input('features-custom', 'value')
)
def update_output(value):
    if value == 'table':
        return generate_table(features_df)
    elif value == 'graph':
        # Normalize the data for better visualization
        features_df_norm = (features_df - features_df.mean()) / features_df.std()
        return dcc.Graph(figure=generate_graph(features_df_norm, ['Power (kW)','Custom'], normalize=True))

# Tab 5 callbacks, model training
@app.callback(
    Output('train-custom-output', 'children'),
    Input('add-custom-feature', 'n_clicks'),  # Triggered when the user adds a custom feature)
    State('custom-feature-output', 'children')  # Triggered when the user adds a custom feature
)
def train_custom_model(n_clicks, custom_features):
    if n_clicks > 0 or custom_features:
        return html.Div([
            html.H3('Train your own model with the custom feature!'),
            html.H4('Select the features you want to use to train the model.'),
            dcc.Dropdown(
                id='features-tools-2',
                options=[
                    {'label': 'Temperature', 'value': 'Temperature (ºC)'},
                    {'label': 'Radiance', 'value': 'Radiance (W/m^2)'},
                    {'label': 'Convolution Feature', 'value': 'Convolution Feature'},
                    {'label': '3h Rolling Average', 'value': '3h Rolling Average'},
                    {'label': 'Power -1h', 'value': 'Power -1h (kW)'},
                    {'label': 'Hour', 'value': 'Hour'},
                    {'label': 'Custom', 'value': 'Custom'}
                ],
                value=[],
                multi=True,
                placeholder='Select features'
            ),
            html.H4('Select the model type.'),
            dcc.RadioItems(
                id='model-2',
                options=[
                    {'label': 'Neural Network', 'value': 'NN'},
                    {'label': 'Random Forest', 'value': 'RF'}
                ],
                value='NN',
            ),
            html.Div(id='train-2'),
            html.Div(id='output-predict-2')
        ])
    return dash.no_update

@app.callback(
    Output('train-2', 'children'),
    Input('model-2', 'value'),
    Input('features-tools-2', 'value'),
    State('n-clicks-store-2', 'data')
)
def selection(model, features, stored_n_clicks):
    if not features or model is None:
        return html.P('Please select at least one feature and the model type to train.', style={'margin':'5px'})

    return html.Div([
        html.Button('Train Model', id='train-button-2', n_clicks=stored_n_clicks),
    ])

@app.callback(
    Output('output-predict-2', 'children'),
    Output('n-clicks-store-2', 'data'),
    Input('train-button-2', 'n_clicks'),
    State('n-clicks-store-2', 'data'),
    State('model-2', 'value'),
    State('features-tools-2', 'value')
)
def train_model(n_clicks, stored_n_clicks, model, features):
    # Ensure the button is pressed and n_clicks is greater than stored_n_clicks
    if n_clicks is None:
        n_clicks = 0  # Handle the case where n_clicks is None initially

    if n_clicks > stored_n_clicks:
        if not features or model is None:
            return html.P('Please select at least one feature and the model type to train.', style={'margin':'5px'}), n_clicks

        # Get the custom feature, and add it to relevant dfs
        custom_vals = features_df['Custom'].values
        df_17_18_cp = df_17_18.copy()
        df_17_18_cp['Custom'] = custom_vals

        X_train = df_17_18_cp[features]
        Y_train = df_17_18_cp['Power (kW)']

        X_test = features_19_df[features]

        if model == 'RF':
            parameters = {'bootstrap': True,
                          'min_samples_leaf': 3,
                          'n_estimators': 200,
                          'min_samples_split': 15,
                          'max_features': 'sqrt',
                          'max_depth': 30,
                          'max_leaf_nodes': None}
            Model = RandomForestRegressor(**parameters)
            Model.fit(X_train.values, Y_train.values)
            Predictions = Model.predict(X_test[features].values)
            df_19_train['Predictions (kW)'] = Predictions
            return html.Div(
                [html.P('Random Forest model trained!'),
                 dcc.Tabs(
                    id='user-forecast-data-2',
                    value='graph',
                    children=[
                        dcc.Tab(label='Graph', value='graph'),
                        dcc.Tab(label='Table', value='table'),
                        dcc.Tab(label='Metrics', value='metrics')
                    ]
                ),
                    html.Div(id='user-forecast-output-2')]
            ), n_clicks

        if model == 'NN':
            Model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=100)
            Model.fit(X_train.values, Y_train.values)
            Predictions = Model.predict(X_test[features].values)
            df_19_train['Predictions (kW)'] = Predictions
            return html.Div(
                [html.P('Neural Network model trained!'),
                 dcc.Tabs(
                        id='user-forecast-data-2',
                        value='graph',
                        children=[
                            dcc.Tab(label='Graph', value='graph'),
                            dcc.Tab(label='Table', value='table'),
                            dcc.Tab(label='Metrics', value='metrics')
                        ]
                    ),
                 html.Div(id='user-forecast-output-2')]
            ), n_clicks

    return dash.no_update, stored_n_clicks

@app.callback(
    Output('user-forecast-output-2', 'children'),
    Input('user-forecast-data-2', 'value')
)
def update_output(value):
    # Get the custom feature, and add it to relevant dfs
    custom_vals = features_19_df['Custom'].values
    df_19_cp = df_19_train.copy()
    df_19_cp['Custom'] = custom_vals

    if value == 'table':
        return generate_table(df_19_cp.drop(['Hour', 'Power -1h (kW)', '3h Rolling Average',
                                          'Convolution Feature', 'Temperature (ºC)',
                                          'Radiance (W/m^2)', 'Custom'], axis=1), style1={'margin': 'auto'},style2={})
    elif value == 'graph':
        return dcc.Graph(figure=generate_graph(df_19_cp, ['Power (kW)','Predictions (kW)']))
    elif value == 'metrics':
        Predictions = df_19_cp['Predictions (kW)']
        return generate_table(metrics_df(Y_test, Predictions,model='Your Model'),style1={'margin': 'auto'},style2={})

if __name__ == '__main__':
    app.run()
