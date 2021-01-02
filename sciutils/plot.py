import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def evaluate_pcolormesh_edges(x, scale='linear'):
    """
    Evaluate the `n + 1` edges of cells for a `pcolormesh` visualisation for `n` cell centroids.

    Parameters
    ----------
    x : np.ndarray
        Centroids of the pcolormesh cells.
    scale : str
        Find the arithmetic midpoints if `linear` and the geometric midpoints if `log`.

    Returns
    -------
    edges : np.ndarray
        Edges of pcolormesh cells.
    """
    if scale == 'log':
        x = np.log(x)
    elif scale != 'linear':
        raise ValueError(scale)

    # Find the (n - 1) midpoints
    midpoints = (x[1:] + x[:-1]) / 2
    # Find the endpoints
    left = 2 * x[0] - midpoints[0]
    right = 2 * x[-1] - midpoints[-1]
    # Construct the edges
    edges = np.concatenate([[left], midpoints, [right]])

    if scale == 'log':
        edges = np.exp(edges)
    return edges


# Define all attributes that should be broadcast
_GEOMETRY_ELEMENTWISE_PROPERTIES = {'linestyle', 'facecolor', 'edgecolor', 'linewidth', 'array'}


def plot_geometry(geometries, aspect='equal', autoscale=True, scale=1, ax=None, **kwargs):
    """
    Plot a shapely geometry using a polygon collection.

    .. note::

       This function does not plot holes in polygons.

    Parameters
    ----------
    geometries :
        Geometry to plot or sequence thereof.
    aspect : str or float, optional
        Aspect ratio of the plot.
    autoscale : bool, optional
        Whether to autoscale the plot.
    ax : optional
        Axes to use for plotting.
    **kwargs : dict
        Keyword arguments passed to `matplotlib.collections.PolyCollection`.

    Returns
    -------
    collection : matplotlib.collections.PolyCollection
        Collection of polygons.
    """
    import shapely.geometry

    ax = ax or plt.gca()

    # If a single geometry is passed, transform it to a list of geometries with one element
    if not isinstance(geometries, collections.abc.Iterable):
        geometries = [geometries]

    # Identify which properties have been provided elementwise
    elementwise_properties = _GEOMETRY_ELEMENTWISE_PROPERTIES & \
        {key for key, value in kwargs.items() if isinstance(value, collections.abc.Iterable)
         and not isinstance(value, str)}

    # Build up all the attributes and vertices
    vertices = []
    collection_kwargs = {}

    for i, geometry in enumerate(geometries):
        if isinstance(geometry, shapely.geometry.MultiPolygon):
            sub_geometries = geometry.geoms
        elif isinstance(geometry, shapely.geometry.Polygon):
            sub_geometries = [geometry]
        else:
            raise ValueError(geometry)

        for geometry in sub_geometries:
            coords = np.asarray(list(geometry.exterior.coords)) * scale
            vertices.append(coords)

            # Deal with elementwise attributes
            for key in elementwise_properties:
                value = kwargs[key]
                collection_kwargs.setdefault(key, []).append(value[i % len(value)])

    array = collection_kwargs.get('array')
    if array is not None:
        collection_kwargs['array'] = np.asarray(array)

    # Copy over remaining kwargs
    kwargs.update(collection_kwargs)
    polys = mpl.collections.PolyCollection(vertices, **kwargs)
    ax.add_collection(polys)

    if aspect:
        ax.set_aspect(aspect)
    if autoscale:
        ax.autoscale_view()
    return polys


def alpha_cmap(color, name=''):
    """
    Create a monochrome colormap that maps scalars to varying transparencies.

    Parameters
    ----------
    color : str, int, or tuple
        Base color to use for the colormap.
    name : str
        Name of the colormap.
    **kwargs : dict
        Keyword arguments passed to :meth:`mpl.colors.LinearSegmentedColormap.from_list`.

    Returns
    -------
    cmap : mpl.colors.Colormap
        Colormap encoding scalars as transparencies.
    """
    if isinstance(color, int):
        color = f'C{color}'
    return mpl.colors.LinearSegmentedColormap.from_list(name, [
        mpl.colors.to_rgba(color, alpha=0.0),
        mpl.colors.to_rgba(color, alpha=1.0),
    ])
