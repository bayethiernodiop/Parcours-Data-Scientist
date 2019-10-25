# -*- coding: utf-8 -*-


# Import applications data
import os
import pandas as pd

dir_name = '../data/cleaned'
file_name = 'data_test.csv'

file_path = os.path.join(dir_name, file_name)

data_test = pd.read_csv(file_path, index_col='SK_ID_CURR')
customer_id = 100001

personal_data =  data_test.loc[customer_id:customer_id]



import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Tableau de Bord',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.H3(children='Prêt à dépenser',
             style={
                 'textAlign': 'center',
                 'color': colors['text']
                 }
    ),
    
    html.H4(children='Numéro de client',
            style={
                'textAlign': 'left',
                'color': colors['text']
            }    
    ),
    
    html.Div([
        dcc.Input(id='my-id', value='100001', type='text'),
        html.Div(id='my-div')
    ]),     
    
    html.H4(children='Données personnelles',
            style={
                'textAlign': 'left',
                'color': colors['text']
            }    
    ),
    
    generate_table(personal_data),
    
    html.H4(children='Scoring',
            style={
                'textAlign': 'left',
                'color': colors['text']
            }
    ),

    dcc.Graph(
        id='example-graph-2',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        }
    ), 
    
])


@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    return 'You\'ve entered "{}"'.format(input_value)

if __name__ == '__main__':
    app.run_server(debug=True)