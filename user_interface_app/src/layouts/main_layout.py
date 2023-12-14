# main_layout.py
import plotly.graph_objects as go
import numpy as np
from dash import html, dcc
import os
from PIL import Image
import random
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils import utils

def create_well_grid(image_path, num_rows, num_columns, size_ratio):
    x_coords = np.tile(np.arange(num_columns), num_rows) * size_ratio
    y_coords = np.repeat(np.arange(num_rows)[::-1], num_columns) * size_ratio


    # Create a blank scatter plot
    fig = go.Figure(data=go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        marker=dict(size=40, opacity=0),  # Make markers invisible
        showlegend=False,
        text = utils.generate_well_annotations(num_rows, num_columns),
        customdata=[f'Well {i+1}' for i in range(len(x_coords))],
        hoverlabel = dict(namelength=0),
        hoverinfo="text"
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

    r = 0.3 * size_ratio
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
        
        fig.add_shape(
        type="rect",
        x0=-1.5 * size_ratio,  # Starting x (adjusted to be slightly outside the first well)
        y0=(num_rows+0.5) * size_ratio,  # Starting y (adjusted to be slightly above the top row)
        x1=(num_columns+0.5) * size_ratio,  # Ending x (adjusted to be slightly outside the last well in a row)
        y1=-1.5 * size_ratio,  # Ending y (adjusted to be slightly below the bottom row)
        line=dict(
            color="Black",
            width=3,
        ),
    )
    
    # Add images for each well
    for x, y in zip(x_coords, y_coords):
        fig.add_layout_image(
            dict(
                source=Image.open(image_path),  # Path to your well image
                xref="x",
                yref="y",
                x=x - 0.75 * size_ratio,  # Adjust for correct alignment
                y=y + 0.75 * size_ratio,  # Adjust for correct alignment
                sizex=1.5 * size_ratio,  # Size of the image in x-axis units
                sizey=1.5 * size_ratio,  # Size of the image in y-axis units
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
    
    fig_width, fig_height = 100 * num_columns * size_ratio, 100 * num_rows * size_ratio
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
        width=1200,  # Width between 400 and max_width
        height=800,  # Height between 400 and max_height
        xaxis=dict(range=[-1, num_columns], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-1, num_rows], showgrid=False, zeroline=False, visible=False, scaleanchor='x', scaleratio=1),
        plot_bgcolor='white'
    )
    fig['layout']['xaxis'].update(autorange = True)
    fig['layout']['yaxis'].update(autorange = True)

    row_labels = [chr(65 + i) for i in range(num_rows)][::-1] 
    for i, label in enumerate(row_labels):
        fig.add_annotation(
            x=-1 * size_ratio,  # Adjust x position for the row labels
            y=i * size_ratio,
            xref='x',
            yref='y',
            text=label,
            showarrow=False,
            yanchor='middle',
            font=dict(
                    size=16*size_ratio,  # Adjust font size
                    color="black",  # Adjust font color
                    family="Arial"
                ),
        )

    # Add column labels (1, 2, 3, ...)
    for i in range(num_columns):
        fig.add_annotation(
            x=i * size_ratio,
            y=num_rows * size_ratio,  # Adjust y position for the column labels
            xref='x',
            yref='y',
            text=str(i + 1),
            showarrow=False,
            xanchor='center',
            font=dict(
                    size=16*size_ratio,  # Adjust font size
                    color="black",  # Adjust font color
                    family="Arial"
                ),
        )


    return fig

def get_layout(image_path, num_rows, num_columns, size_ratio):
    wells = utils.generate_well_annotations(num_rows, num_columns)

    layout = html.Div(
        style={'display': 'flex', 'flex-direction': 'row'},  # Horizontal layout
        children=[
            # Left part - Wells representation (Graph)
            html.Div(
                style={'width': '70%', 'padding': '10px'},  # Adjust width as needed
                children=[
                    html.H1(f"Grid with {num_rows}x{num_columns} Wells"),
                    dcc.Graph(id='well-plot', figure=create_well_grid(image_path, num_rows, num_columns, size_ratio), config={'autosizable': True})
                ]
            ),

            # Right part - Controls and options
            html.Div(
            style={'width': '30%', 'padding': '20px', 'border-left': '2px solid #ddd'},  # Adjusted sidebar style
            children=[
                # Run name input and save button
                html.Div([
                    dcc.Input(id='run-name-input', type='text', placeholder='Enter run name', style={'margin-right': '10px'}),
                    html.Button('Save Run', id='save-run-button')
                ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}),

                # Save status message
                html.Div(id='save-status', style={'margin-bottom': '20px'}),

                # Saved runs dropdown and refresh button
                html.Div([
                    dcc.Dropdown(
                        id='saved-runs-dropdown',
                        options=[{'label': run, 'value': run} for run in utils.list_saved_runs()],
                        placeholder='Select a Run',
                        style={'width': '70%', 'margin-right': '10px'}
                    ),
                    html.Button("Refresh Runs", id="refresh-runs-button")
                ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}),

                # Load run button
                html.Button('Load Run', id='load-run-button', style={'width': '100%', 'margin-bottom': '20px'}),

                html.Div([
                    dcc.Store(id='well-categorizations', storage_type='memory'),
                    dcc.Input(id='selected-wells', style={'display': 'none'})
                ]),

                                # Well selector dropdown
                html.Div([
                    html.Label("Select Wells:", style={'margin-bottom': '5px', 'font-weight': 'bold'}),
                    # In your layout
                    dcc.Dropdown(
                        id='well-selector',
                        options=[{'label': well, 'value': well} for well in wells],
                        value=[],  # Default value or initial selection
                        multi=True,  # Allow multiple selections
                    )
                ]),

                # Categorization controls
                html.Div([
                    html.Label("Categorize as:", style={'font-weight': 'bold'}),
                    dcc.Input(
                        id='well-category',
                        type='text',
                        placeholder='Enter custom annotation',
                        style={'margin-bottom': '20px', 'width': '100%'}  # Full width input field
                    ),
                    html.Button('Update Wells', id='update-button', style={'width': '100%', 'margin-bottom': '20px'})
                ]),
                html.Div(id='categorization-output'),
                html.Div(id='click-data', children=[]),

                # Clear annotations button
                html.Button("Clear Annotations", id="clear-annotations-button", style={'width': '100%', 'margin-bottom': '20px'}),
                # Title of the image
                html.Div(id='image-title', style={'font-size': '20px', 'text-align': 'center', 'font-weight': 'bold', 'margin-top': '5px'}),
                # Image display
                html.Img(id='organoid-image', src='', style={'max-width': '100%', 'height': 'auto', 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
            ]
        )

        # ... [rest of your layout] ...
    ])
    return layout