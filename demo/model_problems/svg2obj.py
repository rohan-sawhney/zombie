#!/usr/local/bin/python3

from argparse import ArgumentParser, ArgumentTypeError
import numpy as np
import os
from shapely.geometry import LinearRing, Polygon
from svgpathtools import svg2paths, Line

def path_to_obj(path, n_points_per_segment):
    '''
    Given an SVG path object, convert the path into an OBJ friendly list of vertices and line segments.
    The indices will be relative to the path and the vertices + line segments will be in their natural
    ordering. Handles closed paths that require a segment from the last to the first vertex.

    Parameters
    -----------
    path: svgpathtools path
    
    n_points_per_segment: int
        the number of points along each non-linear portion of the path (ie existing line segments default to 2 points)
    
    Returns
    --------
    path_v: ndarray
        n_paths x n_path_vertices x 3
    
    path_l: nadarray
        n_paths x n_path_line_segments x 2
    
    attributes: list of dicts
        n_paths dictionaries indicating if path is empty and/or closed
    '''
    v, l = [], []
    props = {
        'is_empty': len(path) == 0,
        'is_closed': False,
    }
    
    if len(path) == 0:
        return np.array(v), np.array(l), props
    
    props['is_closed'] = path[0].start == path[-1].end
    for i, segment in enumerate(path):
        is_last_segment = i == len(path) - 1
            
        n_points = n_points_per_segment
        for j in range(n_points):
            pt = segment.point(j / n_points)
            v.append(np.array([pt.real, pt.imag]))
            if j < n_points - 1:
                l.append(np.array([len(v) - 1, len(v)]))

        if not is_last_segment:
            l.append((len(v) - 1, len(v)))
        elif is_last_segment and not props['is_closed']:
            pt = segment.end
            v.append(np.array([pt.real, pt.imag]))
            l.append(np.array([len(v) - 2, len(v) - 1]))
        else:
            l.append(np.array([len(v) - 1, 0]))

    return np.array(v), np.array(l), props

def normalize(path_v):
    '''
    Normalize vertex locations (in place) across all paths.

    Parameters
    -----------
    path_v: ndarray
        n_paths x n_path_vertices x 3 array holding vertices for each path
    '''
    n_v = reduce(lambda x, path_vi: x + len(path_vi), path_v, 0)

    cm = np.array([0.0, 0.0])
    for i in range(len(path_v)):
        cm += np.mean(path_v[i], axis=0) * (len(path_v[i]) / n_v)

    radius = 0 
    for i in range(len(path_v)):
        path_v[i] -= cm[np.newaxis, :]
        radius = max(radius, np.amax(np.linalg.norm(path_v[i], axis=1)))

    for i in range(len(path_v)):
        path_v[i] /= radius

    return path_v

def orient_curves(path_v, path_l, use_ccw):
    '''
    Create consistent orientation across curves.

    Parameters
    ------------
    path_v: ndarray
        n_paths x n_path_vertices x 3 array holding vertices for each path
    
    path_l: ndarray
        n_paths x n_line_segments x 2 array holding indices for each path

    use_ccw: bool 
        use counter clockwise orienation for all curves
    '''
    n_paths = len(path_v)
    ring = [LinearRing(path_v[i]) for i in range(n_paths)]
    for i in range(n_paths):
        if not (ring[i].is_ccw == use_ccw):
            path_l[i][:, [0, 1]] = path_l[i][:, [1, 0]]

def orient_curves_by_containment_order(path_v, path_l, path_props):
    '''
    Adjust orientation of curves (in place) to allow for a consistent notion signed distance
    relative to line segments. Any curve which completely contains another curve should have
    the opposite orientation. We ignore open curves and we require that all closed curves are 
    non-intersecting.

    Note: assume that path_v and path_l define a curve with vertices in order [(v_0, v_1), (v_1, v_2), ...]
    Note: path_l will be updated in place

    Parameters
    ------------
    path_v: ndarray
        n_paths x n_path_vertices x 3 array holding vertices for each path
    
    path_l: ndarray
        n_paths x n_line_segments x 2 array holding indices for each path

    path_props: array of dicts
        additional information about each path
    '''
    n_paths = len(path_v)

    # primitives
    closed = [path_props[i]['is_closed'] for i in range(n_paths)]
    ring = [LinearRing(path_v[i]) for i in range(n_paths)]
    polygon = [Polygon(path_v[i]) for i in range(n_paths)]
    
    # compute area, ignore open curves
    signed_area = np.array([-polygon[i].area if ring[i].is_ccw else polygon[i].area
                            for i in range(n_paths)])
    signed_area = np.where(closed, signed_area, 0)

    # if using this orientation method, ensure that no curves intersect each other
    for i in range(n_paths):
        for j in range(i+1, n_paths):
            if ring[i].intersects(ring[j]):
                raise Exception('Cannot use auto-orientation with intersecting curves.')

    # contains[i][j] ==> closed curve i completely contains curve j
    contains = np.array([
        [
            (1 if polygon[i].contains_properly(ring[j]) else 0) 
            if closed[i] else 0 
            for j in range(n_paths)
        ]
        for i in range(n_paths)
    ])

    # determine which curves need orientation flipped (i.e. curve orientation identical to containing curve)
    flip_orientation = [False] * n_paths
    curves_largest_to_smallest_area = np.argsort(np.abs(signed_area))[::-1]
    for idx in curves_largest_to_smallest_area:
        containing_curves = np.where(contains[:, idx] == 1)[0]
        if closed[idx] and len(containing_curves) > 0:
            # find parent (smallest area curve that contains this one) ensure orientation is flipped
            parent_idx = containing_curves[np.argsort(signed_area[containing_curves])[0]]
            flip_orientation[idx] = (np.sign(signed_area[parent_idx]) == np.sign(signed_area[idx]))

            if flip_orientation[idx]:
                signed_area[idx] *= -1

    for i in range(n_paths):
        if flip_orientation[i]:
            path_l[i][:, [0, 1]] = path_l[i][:, [1, 0]]

def svg_to_obj(svg_path, obj_path, normalize, use_ccw, auto_orient_curves, n_points_per_segment):
    '''
    Read an SVG file and convert it into an OBJ file.

    Parameters
    -----------
    svg_path: string

    obj_path: string

    normalize: bool

    auto_orient_curves: bool

    n_points_per_segment: int
    '''
    paths, attributes = svg2paths(svg_path)
    paths = [path for path in paths if len(path) > 0]
    path_v, path_l, path_props = [[]] * len(paths), [[]] * len(paths), [{}] * len(paths)
    for i, path in enumerate(paths):
        path_v[i], path_l[i], path_props[i] = path_to_obj(path, n_points_per_segment)

    if normalize:
        path_v = normalize(path_v)

    orient_curves(path_v, path_l, use_ccw)

    if auto_orient_curves:
        orient_curves_by_containment_order(path_v, path_l, path_props)

    v, l = [], []
    for i in range(len(path_v)):
        path_l[i] += len(v) + 1
        l.extend(path_l[i])
        v.extend(path_v[i])

    with open(obj_path, 'w') as obj_file:
        [obj_file.write(f'v {vi[0]} {vi[1]} 0\n') for vi in v]
        [obj_file.write(f'l {li[0]} {li[1]}\n') for li in l]
            
def parse_float_triplet(s):
    try:
        x, y = map(float, s.split(' '))
        return np.array([x, y])

    except ValueError:
        raise ArgumentTypeError("Points must be in the format 'x,y,z'")

if __name__ == '__main__':
    parser = ArgumentParser(description='Convert SVG to OBJ.')
    parser.add_argument('svg_filename', type=str, help='Path to the SVG file')
    parser.add_argument('--out', type=str, nargs='?', help='Path to the output OBJ file')
    parser.add_argument('--normalize', action='store_true', help='Normalize vertex positions')
    parser.add_argument('--use_ccw', action='store_true', help='Use counter-clockwise orientation instead of clockwise')
    parser.add_argument('--auto_orient_curves', action='store_true', help='Orient curves so that they are coherent. Only for non-overlapping, closed curves.')
    parser.add_argument('--n_points_per_segment', type=int, nargs='?',help='Number of points along each segment of an SVG path.', default=10)

    args = parser.parse_args()

    output = os.path.splitext(args.out if args.out is not None else args.svg_filename)[0] + '.obj'
    os.makedirs(os.path.dirname(output), exist_ok=True)

    svg_to_obj(args.svg_filename,
               output,
               args.normalize,
               args.use_ccw,
               args.auto_orient_curves,
               args.n_points_per_segment)
