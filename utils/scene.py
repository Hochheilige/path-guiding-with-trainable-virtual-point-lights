import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import numpy as np
import matplotlib.pyplot as plt


def world_to_ndc(scene, batch):
    """Transforms 3D world coordinates into normalized device coordinates (NDC) using the perspective transformation matrix.

    Args:
        scene (mi.Scene): Mitsuba 3 scene containing the camera information.
        batch (array_like): Array of 3D world coordinates.

    Returns:
        mi.Point3f: Array of 3D points in NDC.
    """
    sensor: mi.Sensor = mi.traverse(scene.sensors()[0])
    fov = sensor['x_fov']
    near = sensor['near_clip']
    far = sensor['far_clip']

    trafo = mi.ProjectiveTransform4f().perspective(fov, near, far)
    to_world_inv = sensor['to_world'].inverse()

    proj_tf = mi.Transform4f(trafo.matrix)
    world_inv_tf = mi.Transform4f(to_world_inv.matrix)
    combined = proj_tf @ world_inv_tf

    pts = combined @ mi.Point3f(np.array(batch.T))
    return pts


def ndc_to_pixel(pts, h, w):
    """Converts points in NDC to pixel coordinates.

    Args:
        pts (mi.Point2f): Points in NDC.
        h (float): Height of the image in pixels.
        w (float): Width of the image in pixels.

    Returns:
        mi.Point2f: Pixel coordinates of the given points.
    """
    hh, hw = h/2, w/2
    return mi.Point2f(dr.fma(pts.x, -hw, hw), dr.fma(pts.y, -hw, hh))  # not typo


def draw_multi_segments(starts, ends, color):
    """Draws multiple line segments on a plot.

    Args:
        starts (mi.Point2f): Starting points of the line segments.
        ends (mi.Point2f): Ending points of the line segments.
        color (str): Color of the line segments.
    """
    a = np.c_[starts.x, starts.y]
    b = np.c_[ends.x, ends.y]
    plt.plot(*np.c_[a, b, a*np.nan].reshape(-1, 2).T, color)


def pix_coord(scene, batch, h, w):
    """Calculates the pixel coordinates of the given 3D world coordinates.

    Args:
        scene (mi.Scene): Mitsuba 3 scene containing the camera information.
        batch (array_like): Array of 3D world coordinates.
        h (float): Height of the image in pixels.
        w (float): Width of the image in pixels.

    Returns:
        mi.Point2f: Pixel coordinates of the given 3D world coordinates.
    """
    return ndc_to_pixel(world_to_ndc(scene, batch), h, w)
