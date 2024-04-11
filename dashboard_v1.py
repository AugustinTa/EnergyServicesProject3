import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px

import load_models as lm
import dashfrance as dfra

#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Initialize fig1 and fig2
data={'Region':['Corse'],'Valeur':[0]}
date_test='2022-07-19 00:00:00'
for region in lm.regions:
    value=lm.predict_next_day_value_cons(region,date=date_test,df_met=lm.df_meteo,df_pow=lm.df_power)
    data['Region'].append(dfra.dico_regions[region])
    data['Valeur'].append(value)
df = pd.DataFrame(data)
fig1=dfra.plot_map(df)

region='Ile-de-France'
Y,X=lm.predict_next_day_cons(region,date=date_test,df_met=lm.df_meteo,df_pow=lm.df_power)
fig2= px.line(x=X, y=Y, title='Consommation énergétique région prochaines 24h en MW')

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server

app.layout = html.Div([
    html.H1('IST Energy Forecast tool (kWh), by Augustin Tachoires'),
    html.P('Representing Data and Forecasting for 2019'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Consumption', value='tab-1'),
        dcc.Tab(label='Production', value='tab-2'),
        dcc.Tab(label='Carbon', value='tab-3')
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H1('IST Energy Forecast tool (kWh), by Augustin Tachoires'),
            dcc.DatePickerSingle(
                id='date-picker',
                date='2022-07-19'
            ),
            html.Button('Plot', id='plot-button', n_clicks=0),
            dcc.Graph(
                id='map-graph',
                figure=fig1
            ),
            html.Div([
                dcc.Dropdown(
                    id='region-dropdown',
                    options=[{'label': region, 'value': region} for region in lm.regions],
                    value='Ile-de-France'
                ),
                html.Button('Plot Region', id='plot-region-button', n_clicks=0)
            ]),
            dcc.Graph(id='region-graph',
                      figure=fig2)
        ])

    elif tab == 'tab-2':
        return html.Div([
            html.H4('IST Electricity Forecast (kWh)')
            ])
        
    elif tab == 'tab-3':
        return html.Div([
            html.H4('IST Raw Data')
        ])
    
@app.callback(
    Output('region-graph', 'figure'),
    [Input('plot-region-button', 'n_clicks')],
    [State('region-dropdown', 'value'),State('date-picker', 'date')]
)
def update_region_graph(n_clicks,region,date):
    if n_clicks > 0:
        date=date+' 00:00:00'
        Y,X=lm.predict_next_day_cons(region,date=date,df_met=lm.df_meteo,df_pow=lm.df_power)
        fig_region= px.line(x=X, y=Y, title='in MW')
        return fig_region
    else:
        return fig2
    

@app.callback(
    Output('map-graph', 'figure'),
    [Input('plot-button', 'n_clicks')],
    [State('date-picker', 'date')]
)

def update_map(n_clicks, date):
    if n_clicks > 0:
        data_cons = {'Region': ['Corse'], 'Valeur': [0]}
        date_test = date+' 00:00:00'
        for region in lm.regions:
            value = lm.predict_next_day_value_cons(region, date=date_test, df_met=lm.df_meteo, df_pow=lm.df_power)
            data_cons['Region'].append(dfra.dico_regions[region])
            data_cons['Valeur'].append(value)
        df = pd.DataFrame(data_cons)
        fig = dfra.plot_map(df)
        return fig
    return fig1  # Retourner la figure initiale si aucune date n'est choisie


if __name__ == '__main__':
    app.run_server()