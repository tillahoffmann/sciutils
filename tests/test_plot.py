import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal
import numpy as np
import pytest
import sciutils.plot as sp
import shapely.geometry


def test_evaluate_pcolormesh_edges_linear():
    x = np.arange(3)
    ex = sp.evaluate_pcolormesh_edges(x, 'linear')
    np.testing.assert_allclose(ex, np.arange(4) - 0.5)


def test_evaluate_pcolormesh_edges_log():
    x = 2 ** np.arange(3)
    ex = sp.evaluate_pcolormesh_edges(x, 'log')
    np.testing.assert_allclose(ex, 2 ** (np.arange(4) - 0.5))


def test_evaluate_pcolormesh_edges_invalid():
    with pytest.raises(ValueError):
        sp.evaluate_pcolormesh_edges([], 'invalid')


@check_figures_equal(extensions=["png"])
def test_plot_geometry(fig_test, fig_ref):
    alpha = 0.5

    # Plot three rectangles
    axr = fig_ref.add_subplot()
    for i in range(3):
        color = mpl.cm.viridis(float(i // 2))
        axr.add_patch(mpl.patches.Rectangle((i, i), 1, 1, facecolor=color, edgecolor='none',
                                            alpha=alpha))
    # Add a patch in the top left
    axr.add_patch(mpl.patches.Rectangle((0, 2), 1, 1, facecolor='C3', edgecolor='none'))
    axr.set_aspect('equal')

    # Do the same rectangles as shapely polygons
    axt = fig_test.add_subplot()
    # Generate the polygons and group them
    vertices = np.asarray([
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1),
    ])
    polygons = [shapely.geometry.Polygon(vertices + i) for i in range(3)]
    polygons = [shapely.geometry.MultiPolygon(polygons[:2]), polygons[2]]
    # Plot with a colormap
    sp.plot_geometry(polygons, array=[0, 1], ax=axt, alpha=alpha)

    # Plot a single polygon
    sp.plot_geometry(shapely.geometry.Polygon(vertices + (0, 2)), facecolor='C3', ax=axt)

    for ax in [axr, axt]:
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)


def test_plot_geometry_invalid():
    with pytest.raises(ValueError):
        sp.plot_geometry(None)


@pytest.mark.parametrize('color', [0, 'k'])
def test_alpha_cmap(color):
    cmap = sp.alpha_cmap(color)
    value = 0.7134
    mapped = cmap(value)
    # Check the colour
    rgb = mpl.colors.to_rgb(f'C{color}' if isinstance(color, int) else color)
    assert rgb == mapped[:3]
    # Check the alpha channel
    alpha = mapped[3]
    assert abs(value - alpha) < 1e-3
