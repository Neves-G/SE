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
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df_17_18 = pd.read_csv('dash-app/data/IST_Civil_Pav_2017_2018.csv')
df_19 = pd.read_csv('dash-app/data/2019Test.csv')
ModelNN = joblib.load('dash-app/models/NN_model.pkl')

# Convert date to datetime
df_17_18['Date_start'] = pd.to_datetime(df_17_18['Date_start'])
df_19['Date'] = pd.to_datetime(df_19['Date'])

# Rename columns
df_17_18.rename(columns={'Date_start': 'Date','Power_kW':'Power (kW)','Hour': 'Hour',
                         'Power-1': 'Power -1h (kW)','Rolling3': '3h Rolling Average',
                         'temp_C': 'Temperature (ºC)','solarRad_W/m2': 'Radiance (W/m^2)',
                         'Convolution': 'Convolution Feature'}, inplace=True)
df_19.rename(columns={'Date': 'Date','Civil (kWh)':'Power (kW)','Hour': 'Hour',
                         'Power-1': 'Power -1h (kW)','Rolling3': '3h Rolling Average',
                         'temp_C': 'Temperature (ºC)','solarRad_W/m2': 'Radiance (W/m^2)',
                         'Convolution': 'Convolution Feature'}, inplace=True)

# Set date to index
df_17_18.set_index('Date', inplace=True)
df_19.set_index('Date', inplace=True)

# Make normalized dfs to plot in the same axis
df_17_18_norm = (df_17_18 - df_17_18.mean()) / df_17_18.std()
df_19_norm = (df_19 - df_19.mean()) / df_19.std()

# Use model on 2019 data
X_test = df_19.drop('Power (kW)', axis=1)
Y_test = df_19['Power (kW)']
PredictionsNN = ModelNN.predict(X_test.values)

# Join predictions to 2019 data
df_19['Predictions'] = PredictionsNN

# Make copy of 2019 data for the user to play with
df_19_train = df_19.copy()

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(dataframe.index.name)] + [html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([html.Td(dataframe.index[i])] + [html.Td(dataframe.iloc[i][col]) for col in dataframe.columns])
            for i in range(min(len(dataframe), max_rows))
        ])
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
    mtrsc = {model: [], 'Reference': [None, None, None, 20, None, 0.05]}
    MAE = metrics.mean_absolute_error(test,predictions) 
    MBE = np.mean(test-predictions)
    MSE = metrics.mean_squared_error(test,predictions)  
    RMSE = np.sqrt(metrics.mean_squared_error(test,predictions))
    cvRMSE = RMSE/np.mean(test)
    NMBE = MBE/np.mean(test)

    mtrsc[model] = [MAE, MBE, MSE, RMSE, cvRMSE, NMBE]
    metrics_df = pd.DataFrame(mtrsc, index=['MAE', 'MBE', 'MSE', 'RMSE', 'cvRMSE', 'NMBE'])
    return metrics_df

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Store(id='n-clicks-store', data=0),  # Add this line
    dcc.Tabs(id='tabs', value='tab-home', children=[
        dcc.Tab(label='Home', value='tab-home'),
        dcc.Tab(label='Training Data', value='tab-1'),
        dcc.Tab(label='Forecast Data', value='tab-2'),
        dcc.Tab(label='Forecast Model', value='tab-3'),
        dcc.Tab(label='Forecast Tools', value='tab-4')
    ]),
    html.H2('IST Civil Building Energy Consumption Dashboard'),
    html.P('Visualization of total electricity consumption'),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-home':
        return html.Div([
            html.Div([
                html.H3('Welcome to the IST Civil Building Energy Consumption Dashboard'),
                html.P('This dashboard provides insights into the energy consumption of the IST Civil Building. '
                       'You can explore the training data, forecast data, and model predictions. '
                       'Use the tabs above to navigate through different sections of the dashboard.'),
                html.P('In the "Training Data" tab, you can view and analyze the historical energy consumption data for 2017 and 2018.'),
                html.P('In the "Forecast Data" tab, you can view the forecasted energy consumption data for 2019.'),
                html.P('In the "Forecast Model" tab, you can view the predictions made by the model.'),
                html.P('In the "Forecast Tools" tab, you can select features and train models to predict energy consumption.')
            ], style={'width': '60%', 'display': 'inline-block', 'justifyContent': 'center'}),
            html.Div([
                html.Img(src='/assets/IST_A_RGB_POS.jpg', style={'height': '350px'})
            ], style={'width': '40%', 'display': 'inline-block', 'textAlign': 'right'})
        ], style={'display': 'flex', 'justifyContent': 'center'})
    
    elif tab == 'tab-1':
        return html.Div([
            dcc.Dropdown(id='17_18-data',
                         options=[
                             {'label': '2017/2018 Data Table', 'value': 'table'},
                             {'label': '2017/2018 Data Graph', 'value': 'graph'}
                         ],
                         value='table',
                         searchable=False
                        ),
            html.Div(id='17_18-data-output')
        ])

    elif tab == 'tab-2':
        return html.Div([
            dcc.Dropdown(id='19-data',
                         options=[
                             {'label': '2019 Data Table', 'value': 'table'},
                             {'label': '2019 Data Graph', 'value': 'graph'}
                         ],
                         value='table',
                         searchable=False
                        ),
             html.Div(id='19-data-output')
        ])
    
    elif tab == 'tab-3':
        return html.Div([
            dcc.Dropdown(id='forecast-data',
                            options=[
                                {'label': '2019 Data Table', 'value': 'table'},
                                {'label': '2019 Data Graph', 'value': 'graph'},
                                {'label': 'Metrics', 'value': 'metrics'}
                            ],
                            value='table',
                            searchable=False
                            ),
            html.Div(id='forecast-data-output'),
        ])       
    
    elif tab == 'tab-4':
        return html.Div([
            html.H3('IST Energy Yearly Consumption (kWh)'),
            html.P('Select the features to predict the energy consumption'),
            dcc.Dropdown(
                id='model',
                options=[
                    {'label': 'Neural Network', 'value': 'NN'},
                    {'label': 'Random Forest', 'value': 'RF'}
                ],
                value=None,
                searchable=False,
                placeholder="Select a model..."
            ),
            dcc.Checklist(
                id='features-tools',
                options=[
                    {'label': 'Temperature', 'value': 'Temperature (ºC)'},
                    {'label': 'Radiance', 'value': 'Radiance (W/m^2)'},
                    {'label': 'Convolution Feature', 'value': 'Convolution Feature'},
                    {'label': '3h Rolling Average', 'value': '3h Rolling Average'},
                    {'label': 'Power -1h', 'value': 'Power -1h (kW)'},
                    {'label': 'Hour', 'value': 'Hour'}
                    # Add customizable feature
                ],
                value=[]
            ),
            html.Button('Train Your Model', id='train-model'),
            html.Div(id='output-predict'),
        ])

@app.callback(
    Output('17_18-data-output', 'children'),
    Input('17_18-data', 'value')
)
def update_output(value):
    if value == 'table':
        return generate_table(df_17_18)
    elif value == 'graph':
        return dcc.Graph(figure=generate_graph(df_17_18_norm, 'all', normalize=True))

@app.callback(
    Output('19-data-output', 'children'),
    Input('19-data', 'value')
)
def update_output(value):
    if value == 'table':
        return generate_table(df_19.drop('Predictions', axis=1))
    elif value == 'graph':
        return dcc.Graph(figure=generate_graph(df_19_norm, 'all', normalize=True))

@app.callback(
    Output('forecast-data-output', 'children'),
    Input('forecast-data', 'value')
)
def update_output(value):
    if value == 'table':
        return generate_table(df_19.drop(['Hour', 'Power -1h (kW)', '3h Rolling Average',
                                          'Convolution Feature', 'Temperature (ºC)', 'Radiance (W/m^2)'], axis=1))
    elif value == 'graph':
        return dcc.Graph(figure=generate_graph(df_19, ['Power (kW)','Predictions']))
    elif value == 'metrics':
        return generate_table(metrics_df(Y_test, PredictionsNN))

@app.callback(
    Output('graph-training', 'figure'),
    Input('features-training', 'value')
)
def update_graph_training(selected_features):
    return generate_graph(df_17_18, selected_features)

@app.callback(
    Output('output-predict', 'children'),
    Output('n-clicks-store', 'data'),  # Add this line
    Input('model', 'value'),
    Input('features-tools', 'value'),
    Input('train-model', 'n_clicks'),
    State('n-clicks-store', 'data')  # Add this line
)
def train_model(model, features, n_clicks, stored_n_clicks):
    if not features or model is None:
        return html.P('Please select at least one feature and the model type to train.'), stored_n_clicks

    if n_clicks and n_clicks > stored_n_clicks:  # Ensure n_clicks is greater than stored_n_clicks
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
            df_19_train['Predictions'] = Predictions
            return html.Div(
                [html.P('Random Forest model trained!'),
                 dcc.Dropdown(id='user-forecast-data',
                              options=[
                                  {'label': 'Select features Table', 'value': 'table'},
                                  {'label': 'Selected features Graph', 'value': 'graph'},
                                  {'label': 'Your model metrics', 'value': 'metrics'}
                              ],
                              value='table',
                              searchable=False
                              ),
                 html.Div(id='user-forecast-output')]
            ), n_clicks

        if model == 'NN':
            Model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=300)
            Model.fit(X_train.values, Y_train.values)
            Predictions = Model.predict(X_test[features].values)
            df_19_train['Predictions'] = Predictions
            return html.Div(
                [html.P('Neural Network model trained!'),
                 dcc.Dropdown(id='user-forecast-data',
                              options=[
                                  {'label': 'Select features Table', 'value': 'table'},
                                  {'label': 'Selected features Graph', 'value': 'graph'},
                                  {'label': 'Your model metrics', 'value': 'metrics'}
                              ],
                              value='table',
                              searchable=False
                              ),
                 html.Div(id='user-forecast-output')]
            ), n_clicks

    return dash.no_update, stored_n_clicks

@app.callback(
    Output('user-forecast-output', 'children'),
    Input('user-forecast-data', 'value')
)
def update_output(value):
    if value == 'table':
        return generate_table(df_19_train.drop(['Hour', 'Power -1h (kW)', '3h Rolling Average',
                                          'Convolution Feature', 'Temperature (ºC)', 'Radiance (W/m^2)'], axis=1))
    elif value == 'graph':
        return dcc.Graph(figure=generate_graph(df_19_train, ['Power (kW)','Predictions']))
    elif value == 'metrics':
        Predictions = df_19_train['Predictions']
        return generate_table(metrics_df(Y_test, Predictions))

if __name__ == '__main__':
    app.run(debug=False)
