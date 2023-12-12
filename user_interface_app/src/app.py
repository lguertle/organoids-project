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
import base64

# Assuming db_utils is a module in the utils package with the resize_image function
from layouts import main_layout
from utils import utils

num_columns = 12
num_rows = 8

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the image
image_path = os.path.join(script_dir, '..', 'assets', 'images', 'well.png')

app.layout = main_layout.get_layout(image_path, num_rows, num_columns)

@app.callback(
    Output('organoid-image', 'src'),
    [Input('well-plot', 'clickData')],
    prevent_initial_call=True
)
def display_organoid_image(clickData):
    # Extract the well number from the clicked data
    well_number = clickData['points'][0]['customdata'].split()[1]
    formatted_well_name = f"s{int(well_number):02d}"
    organoids_path = os.path.join(script_dir, '..', 'assets', 'organoids')
    images_organoid_path = os.listdir(organoids_path)
    for image in images_organoid_path:
        if formatted_well_name in image:
            image_path = os.path.join(organoids_path, image)
            encoded_image = base64.b64encode(open(image_path, 'rb').read())
            return 'data:image/jpg;base64,{}'.format(encoded_image.decode())


@app.callback(
    [Output('well-plot', 'figure'),
     Output('well-categorizations', 'data'),
     Output('save-status', 'children')],
    [Input('update-button', 'n_clicks'),
     Input('clear-annotations-button', 'n_clicks'),
     Input('load-run-button', 'n_clicks'),
     Input('save-run-button', 'n_clicks')],
    [State('run-name-input', 'value'),
     State('well-categorizations', 'data'),
     State('well-plot', 'figure'),
     State('well-selector', 'value'),
     State('well-category', 'value'),
     State('saved-runs-dropdown', 'value')]
)
def handle_buttons(update_clicks, clear_clicks, load_clicks, save_clicks, 
                   run_name, categorizations, current_figure, 
                   selected_wells, category, selected_run):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # Initialize default return values
    fig = go.Figure(current_figure) if current_figure else None
    status_message = ""

    if button_id == 'clear-annotations-button':
        # Clear annotations and reset categorizations
        fig.layout["annotations"] = []
        categorizations = {}
        status_message = "Categorizations cleared."
        return fig, categorizations, status_message

    elif button_id == 'update-button':
        if not categorizations:
            categorizations = {}
        if selected_wells and category:
            for well in selected_wells:
                categorizations[well] = category
            status_message = "Categorizations updated."
        else:
            status_message = "No wells or category selected."

    elif button_id == 'load-run-button':
        if selected_run:
            selected_run = os.path.join("runs", selected_run)
            with open(selected_run, 'r') as file:
                categorizations = json.load(file)
            status_message = f"Run '{selected_run}' loaded successfully."

    elif button_id == 'save-run-button':
        if run_name:
            filename = f"runs/{run_name}.json"
            with open(filename, "w") as file:
                json.dump(categorizations, file)
            status_message = f"Run '{run_name}' saved successfully."

    if categorizations and fig:
        for well, category in categorizations.items():
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

    return fig, categorizations, status_message

@app.callback(
    Output('saved-runs-dropdown', 'options'),
    [Input('refresh-runs-button', 'n_clicks')],
    prevent_initial_call=True
)
def refresh_runs_dropdown(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    # Assuming your saved runs are in a specific directory or accessible via a function
    return [{'label': run, 'value': run} for run in utils.list_saved_runs()]

@app.callback(
    Output('image-title', 'children'),
    [Input('well-plot', 'clickData')]  # Replace with your actual trigger
)
def update_image_and_title(clickData):
    well_info = None
    if clickData is not None and 'points' in clickData and clickData['points']:
        well_info = clickData['points'][0].get('text', 'No text available')

    return well_info

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
