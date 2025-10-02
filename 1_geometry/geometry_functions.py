import numpy as np
import pandas as pd
from shapely import LineString, get_coordinates

def create_wiggly_line(p0, p1, n=100, amp=5):
    """
    Creates a wiggly line

    Parameters
    ----------
    p0
        First point as tuple (x, y)
    p1
        End point as tuple (x, y)
    n
        Number of segments
    amp
        amplitude

    Returns
    -------
    linestring
        Shapely LineString
    """

    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy)
    if L == 0:
        raise ValueError("Points must be different")

    # Unit direction and perpendicular vectors
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux # normal

    # parameter t along the line
    t = np.linspace(0, 1, n)

    # base straight line
    xs = x0 + t * dx
    ys = y0 + t * dy

    # wiggle offset perpendicular to the line
    wiggle = amp * np.sin(2 * np.pi * t * 3) # 3 cycles
    xs += wiggle * nx
    ys += wiggle * ny

    return LineString(np.column_stack([xs, ys]))

def total_bearing_change_planar(gdf, geom_col="geometry"):
    """
    Sum of absolute bearing changes (degrees) along each LineString, assuming a projected CRS.
    Bearing is measured clockwise from North (0..360). Turns use the smallest angle (0..180).

    Parameters
    ----------
    gdf
        GeoPandas GeoDataFrame of LineStrings    
    geom_col
        Name of the geometry column

    Returns
    -------
    total absolute bearing change
        Pandas Series
    """
    # Pull all coordinates in one shot, together with their parent row index
    coords, parent_ix = get_coordinates(gdf[geom_col].values, return_index=True)  # (N,2), (N,)
    if coords.shape[0] < 2:
        return pd.Series(0.0, index=gdf.index)

    # Valid consecutive coordinate pairs are those that stay in the same geometry
    same_geom_pair = parent_ix[:-1] == parent_ix[1:]
    if not np.any(same_geom_pair):
        return pd.Series(0.0, index=gdf.index)

    # Build segment endpoints (x1,y1) -> (x2,y2)
    c0 = coords[:-1][same_geom_pair]
    c1 = coords[1:][same_geom_pair]
    seg_parent = parent_ix[:-1][same_geom_pair]

    dx = c1[:, 0] - c0[:, 0]
    dy = c1[:, 1] - c0[:, 1]

    # Drop zero-length segments to avoid NaNs
    nz = (dx != 0) | (dy != 0)
    if not np.any(nz):
        return pd.Series(0.0, index=gdf.index)
    dx, dy, seg_parent = dx[nz], dy[nz], seg_parent[nz]

    # Bearing clockwise from North: arctan2(dx, dy) -> degrees 0..360
    bearings = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

    # Bearing changes are between adjacent segments within each geometry
    same_geom_adjacent_segment = seg_parent[:-1] == seg_parent[1:]
    if not np.any(same_geom_adjacent_segment):
        # Single-segment LineStrings -> zero turn
        out = pd.Series(0.0, index=gdf.index)
        return out

    b0 = bearings[:-1][same_geom_adjacent_segment]
    b1 = bearings[1:][same_geom_adjacent_segment]
    parents_for_turn = seg_parent[1:][same_geom_adjacent_segment]

    # Smallest absolute angle difference in [0, 180]
    # (wrap-around-safe)
    turn = np.abs(((b1 - b0 + 180.0) % 360.0) - 180.0)

    # Aggregate per geometry index
    s = pd.Series(turn).groupby(pd.Series(parents_for_turn)).sum()

    # Return aligned to gdf index (missing -> 0)
    out = pd.Series(0.0, index=gdf.index, dtype=float)
    out.loc[s.index.map(lambda i: gdf.index[i])] = s.values
    return out
