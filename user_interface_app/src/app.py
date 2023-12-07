import os
import sys
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input
import plotly.graph_objects as go
import numpy as np
from PIL import Image

# Assuming db_utils is a module in the utils package with the resize_image function
from layouts import main_layout
from utils import utils

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
script_dir = os.path.dirname(os.path.abspath(__file__))

# Generate coordinates for the wells
num_rows = 4
num_columns = 4
x_coords = np.repeat(np.arange(num_columns), num_rows)
y_coords = np.tile(np.arange(num_rows), num_columns)

# Load the image
image_path = os.path.join(script_dir, '..', 'assets', 'images', 'well.png')
well_image = Image.open(image_path)

# Define your scatter plot
fig = go.Figure(go.Scatter(
    x=x_coords,  # X-coordinates of your wells
    y=y_coords,  # Y-coordinates of your wells
    mode='markers',
    marker=dict(size=20, opacity=0),  # Set markers as invisible but clickable
    text=[f'Well {i+1}' for i in range(len(x_coords))]  # Label each well
))

fig.update_layout(
    clickmode='event+select',
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False)
)

app.layout = main_layout.get_layout(image_path)

# Callback to handle clicks on the plot
@app.callback(
    Output('click-data', 'children'),
    Input('well-plot', 'clickData')
)

def display_click_data(clickData):
    if clickData is not None and 'points' in clickData and clickData['points']:
        well_info = clickData['points'][0].get('text', 'No text available')
        return f"You clicked on {well_info}"
    return "Click on a well to get information"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
