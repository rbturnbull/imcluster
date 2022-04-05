from jinja2 import Environment, PackageLoader, select_autoescape
from collections import defaultdict
from .io import ImclusterIO


def write_html(imcluster_io:ImclusterIO, output_html=None, force:bool=False):

    env = Environment(
        loader=PackageLoader("imcluster"),
        autoescape=select_autoescape()
    )

    template = env.get_template("clusters.html")
    # template = env.get_template("vtab.html")

    if not output_html:
        output_html = "output-clusters.html"

    data = defaultdict(list)
    df = imcluster_io.df.sort_values("cluster")
    clusters = df["cluster"]
    thumbnails = df["thumbnail"]
    filenames = df["filenames"]
    for filename, cluster, thumbnail in zip(filenames, clusters, thumbnails):
        data[cluster].append(dict(filename=filename, thumbnail=thumbnail))

    result = template.render(data=data)

    with open(output_html, 'w') as f:
        f.write(result)