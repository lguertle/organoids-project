# main_layout.py
import plotly.graph_objects as go
import numpy as np
from dash import html, dcc
import os
from PIL import Image
import random
from utils import utils

def create_well_grid(image_path, num_rows, num_columns):
    # Generate coordinates for the wells in a 4x4 grid
    x_coords = np.tile(np.arange(num_columns), num_rows)
    y_coords = np.repeat(np.arange(num_rows)[::-1], num_columns)  # Invert y-axis


    # Create a blank scatter plot
    fig = go.Figure(data=go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        marker=dict(size=40, opacity=0),  # Make markers invisible
        showlegend=False,
        text=[f'Well {i+1}' for i in range(len(x_coords))],
        customdata=[f'Well {i+1}' for i in range(len(x_coords))]
    ))

    labels = ["dead", "keep2", "reseed1", "split", "empty", "keep0", "keep1", "reseed0"]

    # Create a mapping of labels to colors
    label_color_mapping = {
        "dead" : "darkorchid",
        "keep2": "green",
        "reseed1": "blue",
        "split": "orange",
        "empty": "grey",
        "keep0": "red",
        "keep1": "purple",
        "reseed0": "yellow"
    }

    # Randomly assign labels to each well
    well_labels = [random.choice(labels) for _ in x_coords]

    r = 0.2
    line_thickness = 4
    for (x, y, label) in zip(x_coords, y_coords, well_labels):
        color = label_color_mapping[label]
        fig.add_shape(
            dict(type="circle",
                xref="x", yref="y",
                x0=x - r, y0=y - r,
                x1=x + r, y1=y + r,
                line_color=color,
                line=dict(
                 color=color,
                 width=line_thickness  # Set the line thickness here
                ),
                showlegend=False
            )
        )

        # Add invisible markers for hover info
        fig.add_trace(go.Scatter(
            x=[x], 
            y=[y],
            text=[label],  # Set the hover text to the well label
            mode='markers',
            marker=dict(color=color, size=1, opacity=0),  # Invisible marker
            hoverinfo='text',
            showlegend=False
        )
    )

    # Add images for each well
    for x, y in zip(x_coords, y_coords):
        fig.add_layout_image(
            dict(
                source=Image.open(image_path),  # Path to your well image
                xref="x",
                yref="y",
                x=x - 0.5,  # Adjust for correct alignment
                y=y + 0.5,  # Adjust for correct alignment
                sizex=1,  # Size of the image in x-axis units
                sizey=1,  # Size of the image in y-axis units
                sizing="contain",
                layer="below"
            )
        )

    
    # Add a scatter trace for each unique label to create legend entries
    for label, color in label_color_mapping.items():
        fig.add_trace(go.Scatter(
            x=[None],  # No actual data points
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=label,  # The label that will appear in the legend
            showlegend=True
        ))
    
    fig_width, fig_height = 100 * num_columns, 100 * num_rows
    # Update the layout for the legend
    fig.update_layout(
        margin=dict(r=160),  # Adjust right margin to fit legend
        legend=dict(
            title='Labels',  # Legend title
            x=1.05,  # Position the legend outside the plot
            xanchor='left',  # Anchor the legend to the left of the specified position
            y=1,
            yanchor='top',  # Anchor the legend to the top of the specified position
            bgcolor='rgba(255,255,255,0.8)',  # Optional: Set a background color for the legend
            bordercolor='Black',
            borderwidth=2,
        ),
        width=max(fig_width, 400),  # Set a minimum figure width
        height=max(fig_height, 400),  # Set a minimum figure height
        xaxis=dict(range=[-1, num_columns], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-1, num_rows], showgrid=False, zeroline=False, visible=False, scaleanchor='x', scaleratio=1),
        plot_bgcolor='white'
    )


    return fig

def get_layout(image_path, num_rows, num_columns):
    wells = [f"Well {i}" for i in range(1, num_rows*num_columns+1)]

    layout = html.Div(
        style={'display': 'flex', 'flex-direction': 'row'},  # Horizontal layout
        children=[
            # Left part - Wells representation (Graph)
            html.Div(
                style={'width': '70%', 'padding': '10px'},  # Adjust width as needed
                children=[
                    html.H1(f"Grid with {num_rows}x{num_columns} Wells"),
                    dcc.Graph(id='well-plot', figure=create_well_grid(image_path, num_rows, num_columns), config={'autosizable': True})
                ]
            ),

            # Right part - Controls and options
            html.Div(
                style={'width': '30%', 'padding': '10px', 'border-left': '2px solid #ddd'},  # Sidebar style
                children=[
                    html.Div([
                    "Number of Rows: ",
                    dcc.Input(id='num-rows-input', type='number', value=10, min=1)  # Default value set to 10
                    ]),

                    # Input for number of columns
                    html.Div([
                        "Number of Columns: ",
                        dcc.Input(id='num-columns-input', type='number', value=10, min=1)  # Default value set to 10
                    ]),

                    # Button to update the plot
                    html.Button('Update Plot', id='update-plot-button'),
                    dcc.Dropdown(
                    id='well-selector',
                    options=[{'label': well, 'value': well} for well in wells],
                    value=[],  # Default value or initial selection
                    multi=True  # Allow multiple selections
                    ),
                    html.Div([
                    dcc.Input(id='run-name-input', type='text', placeholder='Enter run name'),
                    html.Button('Save Run', id='save-run-button'),
                    ]),
                    html.Div(id='save-status'),
                    html.Div([
                    dcc.Dropdown(
                        id='saved-runs-dropdown',
                        options=[{'label': run, 'value': run} for run in utils.list_saved_runs()],
                        placeholder='Select a Run'
                    ),
                    html.Button('Load Run', id='load-run-button'),
                    ]),
                    dcc.Store(id='well-categorizations', storage_type='memory'),
                    dcc.Input(id='selected-wells', style={'display': 'none'}),
                    html.Label("Categorize as:"),
                    dcc.RadioItems(
                        id='well-category',
                        options=[
                            {'label': 'Control', 'value': 'control'},
                            {'label': 'Treated', 'value': 'treated'}
                        ]
                    ),
                    html.Button('Update Wells', id='update-button'),
                    html.Div(id='categorization-output'),
                    html.Div(id='click-data', children=[])
                    ]
            )
        ]
    )
    return layout




"""
    layout = html.Div([
        html.H1(f"Grid with {num_rows}x{num_columns} Wells"),
        dcc.Graph(id='well-plot', figure=create_well_grid(image_path, num_rows, num_columns), config={'autosizable': True}),
        dcc.Dropdown(
        id='well-selector',
        options=[{'label': well, 'value': well} for well in wells],
        value=[],  # Default value or initial selection
        multi=True  # Allow multiple selections
        ),
        html.Div([
        dcc.Input(id='run-name-input', type='text', placeholder='Enter run name'),
        html.Button('Save Run', id='save-run-button'),
        ]),
        html.Div(id='save-status'),
        html.Div([
        dcc.Dropdown(
            id='saved-runs-dropdown',
            options=[{'label': run, 'value': run} for run in utils.list_saved_runs()],
            placeholder='Select a Run'
        ),
        html.Button('Load Run', id='load-run-button'),
        ]),
        dcc.Store(id='well-categorizations', storage_type='memory'),
        dcc.Input(id='selected-wells', style={'display': 'none'}),
        html.Label("Categorize as:"),
        dcc.RadioItems(
            id='well-category',
            options=[
                {'label': 'Control', 'value': 'control'},
                {'label': 'Treated', 'value': 'treated'}
            ]
        ),
        html.Button('Update Wells', id='update-button'),
        html.Div(id='categorization-output'),
        html.Div(id='click-data', children=[])
    ])
"""
