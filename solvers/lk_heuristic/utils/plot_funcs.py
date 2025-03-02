import os
import plotly.graph_objects as go
from lk_heuristic.utils.io_funcs import import_tsp_file

# get the directory of this file and setup the "plots" dir
file_dir = os.path.dirname(os.path.realpath(__file__))
plots_dir = os.path.abspath(os.path.join(file_dir, "../", "plots"))


def plot_tsp_2d(tsp_file, tour_type="cycle"):
    """
    Plot a tsp tour in a 2D graph using plotly library and save the plot at "plots" folder. 

    :param tsp_file: the tsp file to be plot
    :type tour: str
    :param tour_type: the type of the tour (either 'path' or 'cycle')
    :type tour_type: str
    """

    # get the file name and extension
    file_name, file_ext = os.path.splitext(os.path.basename(tsp_file))

    # import the tsp file
    tsp_header, nodes = import_tsp_file(tsp_file)

    # create an empty figure
    fig = go.Figure()

    # set up layout configs
    # title as tsp name and centered
    # fig template scheme
    fig.update_layout(title={
        'text': tsp_header["NAME"],
        'y': 0.95,
        'x': 0.5},
        template="plotly_white")

    # remove grid lines
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    # lists to colect the plotting data in 2D
    x_coords = []
    y_coords = []

    # loop through each tour node
    for node in nodes:
        # append the x and y coords
        x_coords.append(node.x)
        y_coords.append(node.y)

    # append last edge to close the tour, if 'cycle' tour type
    if tour_type == "cycle":
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])

    # plot the line between two nodes
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='lines',
        line=dict(color="rgb(192,192,192)",
                  width=2),
        showlegend=False))

    # plot the scatter of current node
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        marker=dict(size=5, color="rgb(255,255,255)",
                    line=dict(color="rgb(11,11,64)",
                              width=1),),
        showlegend=False))

    # export the plot into a html file
    fig.write_html(os.path.join(plots_dir, f"{file_name}.html"))


def plot_tsp_3d(tsp_file, tour_type="cycle"):
    """
    Plot a tsp tour in a 3D graph using plotly library and save the plot at "plots" folder

    :param tsp_file: the tsp file to be plot
    :type tour: str
    :param tour_type: the type of the tour (either 'path' or 'cycle')
    :type tour_type: str
    """

    # get the file name and extension
    file_name, file_ext = os.path.splitext(os.path.basename(tsp_file))

    # import the tsp file
    tsp_header, nodes = import_tsp_file(tsp_file)

    # create an empty figure
    fig = go.Figure()

    # set up layout configs
    # title as tsp name and centered
    # fig template scheme
    fig.update_layout(title={
        'text': tsp_header["NAME"],
        'y': 0.95,
        'x': 0.5},
        template="plotly_white")

    # remove grid lines
    fig.update_layout(
        scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    # lists to colect the plotting data in 3D
    x_coords = []
    y_coords = []
    z_coords = []

    # loop through each tour node
    for node in nodes:
        # append the x, y and z coords
        x_coords.append(node.x)
        y_coords.append(node.y)
        z_coords.append(node.z)

    # append last edge to close the tour, if 'cycle' tour type
    if tour_type == "cycle":
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        z_coords.append(z_coords[0])

    # plot the line between two nodes
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='lines',
        line=dict(color="rgb(192,192,192)",
                  width=2),
        showlegend=False))

    # plot the scatter of current node
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers',
        marker=dict(size=5, color="rgb(255,255,255)",
                    line=dict(color="rgb(11,11,64)",
                              width=1),),
        showlegend=False))

    # export the plot into a html file
    fig.write_html(os.path.join(plots_dir, f"{file_name}.html"))
