from bokeh.plotting import figure, output_file, show
from rich.console import Console
console = Console()

from .io import ImclusterIO

def plot(imcluster_io:ImclusterIO, output_html=None, width=1200, height=700, size=8, color="navy", alpha=0.5):
    """
    plot the principle components with tooltips showing the images    
    """

    if not output_html:
        output_html = imcluster_io.output.with_suffix('.html')

    output_file(output_html)
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("desc", "@desc"),
    ]

    p = figure(width=width, height=height, tooltips=TOOLTIPS)
    
    p.circle(imcluster_io.get_column("pca0"), imcluster_io.get_column("pca1"), size=size, color=color, alpha=alpha)
    show(p)
