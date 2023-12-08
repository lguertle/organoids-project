# main_layout.py
import plotly.graph_objects as go
import numpy as np
from dash import html, dcc
import os
from PIL import Image
import random

def create_well_grid(image_path):
    # Generate coordinates for the wells in a 4x4 grid
    num_rows = 4
    num_columns = 4
    x_coords = np.repeat(np.arange(num_columns), num_rows)
    y_coords = np.tile(np.arange(num_rows), num_columns)

    # Create a blank scatter plot
    fig = go.Figure(data=go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        marker=dict(size=40, opacity=0),  # Make markers invisible
        showlegend=False
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
        print("Image path:", os.path.join(image_path))
        print("x, y coordinates:", list(zip(x_coords, y_coords)))
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
        xaxis=dict(range=[-1, 4], showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-1, 4], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='white',
        width=800,
        height=800,
    )

    return fig

def get_layout(image_path):
    layout = html.Div([
        html.H1("Rectangle with 4x4 Wells"),
        dcc.Graph(id='well-plot', figure=create_well_grid(image_path)),
        html.Div(id='click-data', children=[])
    ])

    return layout
