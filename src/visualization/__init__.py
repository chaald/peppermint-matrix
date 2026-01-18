import matplotlib
from IPython.display import HTML, display
from src.utils import is_notebook

def set_themes(theme="grey"):
    if theme == "grey":
        matplotlib.rcParams['figure.figsize'] = [8.32, 6.24]
        matplotlib.rcParams['figure.facecolor'] = "grey"
        matplotlib.rcParams['axes.facecolor'] = "lightgrey"
        matplotlib.rcParams['scatter.edgecolors'] = "black"
        
        # Set ipywidget background for matplotlib widget mode
        if is_notebook():
            display(HTML("""
            <style>
                .jupyter-widgets.widget-inline-hbox,
                .jupyter-widgets.widget-box,
                .jupyter-matplotlib-canvas,
                .jupyter-matplotlib,
                .widget-output,
                .jp-OutputArea-output,
                .jp-OutputArea-child,
                .jp-OutputArea,
                .jp-Cell-outputWrapper,
                div.output_subarea,
                div.output_area,
                .cell-output-ipywidget-background {
                    background-color: grey !important;
                    padding: 0 !important;
                }
                
                /* Target the canvas container specifically */
                canvas.jupyter-matplotlib-canvas {
                    background-color: grey !important;
                }
            </style>
            """))
    else:
        raise Exception(f"Theme \"{theme}\" is undefined.")