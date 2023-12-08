import os
import sys
import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go
import numpy as np
from PIL import Image
from dash.exceptions import PreventUpdate
import json
from dash import callback_context

# Assuming db_utils is a module in the utils package with the resize_image function
from layouts import main_layout
from utils import utils

num_columns = 10
num_rows = 10

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the image
image_path = os.path.join(script_dir, '..', 'assets', 'images', 'well.png')

app.layout = main_layout.get_layout(image_path, num_rows, num_columns)

@app.callback(
    Output('click-data', 'children'),
    [Input('well-plot', 'clickData')])
def display_click_data(clickData):
    if clickData is not None and 'points' in clickData and clickData['points']:
        well_info = clickData['points'][0].get('text', 'No text available')
        return f"You clicked on {well_info}"
    return "Click on a well to get information"


@app.callback(
    Output('well-plot', 'figure'),
    Input('well-categorizations', 'data'),
    State('well-plot', 'figure'))
def update_plot(categorizations, current_figure):
    if categorizations is None:
        raise PreventUpdate

    # Create a new figure from the current figure
    fig = go.Figure(current_figure)

    # Loop through categorizations and add/update text annotations on the plot
    for well, category in categorizations.items():
        # Extract the index from the well name
        well_index = int(well.split()[1]) - 1  # Convert to 0-based index

        # Calculate the row and column for this well
        row = well_index // num_columns
        col = well_index % num_columns

        # Determine the position of the well on the plot
        x_position = col
        y_position = num_rows - 1 - row  # Subtract from num_rows - 1 to invert y-axis

        # Add or update annotation
        fig.add_annotation(
            x=x_position, 
            y=y_position, 
            text=category, 
            showarrow=False, 
            yshift=30,  # Adjust yshift for positioning
            font=dict(
                size=12,  # Adjust font size
                color="black"  # Adjust font color
            ),
            align="center",  # Align text
            bgcolor="white",  # Background color of annotation
            bordercolor="black",  # Border color
            borderpad=2,  # Border padding
        )

    return fig

@app.callback(
    Output('save-status', 'children'),  # You can add an output to show save status
    [Input('save-run-button', 'n_clicks')],
    [State('run-name-input', 'value'),
     State('well-categorizations', 'data')]
)
def save_run(n_clicks, run_name, categorizations):
    if n_clicks is None or not run_name:
        raise PreventUpdate

    # Save the categorizations with the run name
    filename = f"{run_name}.json"  # Create a filename from the run name
    with open(filename, "w") as file:
        json.dump(categorizations, file)

    return f"Run '{run_name}' saved successfully!"

@app.callback(
    Output('well-categorizations', 'data'),
    [Input('update-button', 'n_clicks'),  
     Input('load-run-button', 'n_clicks')],
    [State('well-selector', 'value'), 
     State('well-category', 'value'),  # Use 'well-category' instead of 'category-selector'
     State('saved-runs-dropdown', 'value'),
     State('well-categorizations', 'data')]
)
def combined_callback(update_btn_clicks, load_btn_clicks, selected_wells, category, selected_run, current_data):
    ctx = callback_context

    # Determine which input triggered the callback
    if not ctx.triggered:
        raise PreventUpdate
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'update-button' and selected_wells is not None and category is not None:
        # Logic for updating categorizations
        if current_data is None:
            current_data = {}
        for well in selected_wells:
            current_data[well] = category
        return current_data

    elif trigger_id == 'load-run-button' and selected_run is not None:
        # Logic for loading a run
        with open(selected_run, 'r') as file:
            loaded_data = json.load(file)
        return loaded_data

    raise PreventUpdate


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
