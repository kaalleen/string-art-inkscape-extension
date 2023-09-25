# Methods from the ski-image library
# source: https://github.com/scikit-image/scikit-image/blob/main/skimage/draw/_draw.pyx

from math import cos, fabs, pi, sin, sqrt

import numpy as np


def _coords_inside_image(rr, cc, shape, val=None):
    """
    Return the coordinates inside an image of a given shape.

    Parameters
    ----------
    rr, cc : (N,) ndarray of int
        Indices of pixels.
    shape : tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates.  Must be at least length 2. Only the first two values
        are used to determine the extent of the input image.
    val : (N, D) ndarray of float, optional
        Values of pixels at coordinates ``[rr, cc]``.

    Returns
    -------
    rr, cc : (M,) array of int
        Row and column indices of valid pixels (i.e. those inside `shape`).
    val : (M, D) array of float, optional
        Values at `rr, cc`. Returned only if `val` is given as input.
    """
    mask = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    if val is None:
        return rr[mask], cc[mask]
    else:
        return rr[mask], cc[mask], val[mask]


def ellipse_perimeter(r_o, c_o, r_radius, c_radius, orientation=0, shape=None):
    """Generate ellipse perimeter coordinates.

    Parameters
    ----------
    r_o, c_o : int
        Centre coordinate of ellipse.
    r_radius, c_radius : int
        Minor and major semi-axes. ``(r/r_radius)**2 + (c/c_radius)**2 = 1``.
    orientation : cnp.float64_t
        Major axis orientation in clockwise direction as radians.
    shape : tuple
        Image shape which is used to determine the maximum extent of output pixel
        coordinates. This is useful for ellipses that exceed the image size.
        If None, the full extent of the ellipse is used.

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the ellipse perimeter.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    References
    ----------
    .. [1] A Rasterizing Algorithm for Drawing Curves, A. Zingl, 2012
           http://members.chello.at/easyfilter/Bresenham.pdf
    """

    # If both radii == 0, return the center to avoid infinite loop in 2nd set
    if r_radius == 0 and c_radius == 0:
        return np.array(r_o), np.array(c_o)

    # Pixels
    rr = list()
    cc = list()

    # Compute useful values
    rd = r_radius * r_radius
    cd = c_radius * c_radius

    r = c = e2 = err = None

    ir0 = ir1 = ic0 = ic1 = ird = icd = None
    sin_angle = ra = ca = za = a = b = None

    if orientation == 0:
        c = -c_radius
        r = 0
        e2 = rd
        err = c * (2 * e2 + c) + e2
        while c <= 0:
            # Quadrant 1
            rr.append(r_o + r)
            cc.append(c_o - c)
            # Quadrant 2
            rr.append(r_o + r)
            cc.append(c_o + c)
            # Quadrant 3
            rr.append(r_o - r)
            cc.append(c_o + c)
            # Quadrant 4
            rr.append(r_o - r)
            cc.append(c_o - c)
            # Adjust `r` and `c`
            e2 = 2 * err
            if e2 >= (2 * c + 1) * rd:
                c += 1
                err += (2 * c + 1) * rd
            if e2 <= (2 * r + 1) * cd:
                r += 1
                err += (2 * r + 1) * cd
        while r < r_radius:
            r += 1
            rr.append(r_o + r)
            cc.append(c_o)
            rr.append(r_o - r)
            cc.append(c_o)

    else:
        sin_angle = sin(orientation)
        za = (cd - rd) * sin_angle
        ca = sqrt(cd - za * sin_angle)
        ra = sqrt(rd + za * sin_angle)

        a = ca + 0.5
        b = ra + 0.5
        za = za * a * b / (ca * ra)

        ir0 = int(r_o - b)
        ic0 = int(c_o - a)
        ir1 = int(r_o + b)
        ic1 = int(c_o + a)

        ca = ic1 - ic0
        ra = ir1 - ir0
        za = 4 * za * cos(orientation)
        w = ca * ra
        if w != 0:
            w = (w - za) / (w + w)
        icd = int(floor(ca * w + 0.5))
        ird = int(floor(ra * w + 0.5))

        # Draw the 4 quadrants
        rr_t, cc_t = _bezier_segment(ir0 + ird, ic0, ir0, ic0, ir0, ic0 + icd, 1-w)
        rr.extend(rr_t)
        cc.extend(cc_t)
        rr_t, cc_t = _bezier_segment(ir0 + ird, ic0, ir1, ic0, ir1, ic1 - icd, w)
        rr.extend(rr_t)
        cc.extend(cc_t)
        rr_t, cc_t = _bezier_segment(ir1 - ird, ic1, ir1, ic1, ir1, ic1 - icd, 1-w)
        rr.extend(rr_t)
        cc.extend(cc_t)
        rr_t, cc_t = _bezier_segment(ir1 - ird, ic1, ir0, ic1, ir0, ic0 + icd,  w)
        rr.extend(rr_t)
        cc.extend(cc_t)

    if shape is not None:
        return _coords_inside_image(np.array(rr, dtype=np.intp),
                                    np.array(cc, dtype=np.intp), shape)
    return np.array(rr, dtype=np.intp), np.array(cc, dtype=np.intp)


def line_aa(r0, c0, r1, c1):
    """Generate anti-aliased line pixel coordinates.
    Source: https://github.com/scikit-image/scikit-image/blob/main/skimage/draw/_draw.pyx

    Parameters
    ----------
    r0, c0 : int
        Starting position (row, column).
    r1, c1 : int
        End position (row, column).

    Returns
    -------
    rr, cc, val : (N,) ndarray (int, int, float)
        Indices of pixels (`rr`, `cc`) and intensity values (`val`).
        ``img[rr, cc] = val``.

    References
    ----------
    .. [1] A Rasterizing Algorithm for Drawing Curves, A. Zingl, 2012
           http://members.chello.at/easyfilter/Bresenham.pdf
    """
    rr = list()
    cc = list()
    val = list()

    dc = abs(c0 - c1)
    dr = abs(r0 - r1)
    err = dc - dr

    err_prime = c = r = sign_c = sign_r = ed = None

    if c0 < c1:
        sign_c = 1
    else:
        sign_c = -1

    if r0 < r1:
        sign_r = 1
    else:
        sign_r = -1

    if dc + dr == 0:
        ed = 1
    else:
        ed = sqrt(dc*dc + dr*dr)

    c, r = c0, r0
    while True:
        cc.append(c)
        rr.append(r)
        val.append(fabs(err - dc + dr) / ed)

        err_prime = err
        c_prime = c

        if (2 * err_prime) >= -dc:
            if c == c1:
                break
            if (err_prime + dr) < ed:
                cc.append(c)
                rr.append(r + sign_r)
                val.append(fabs(err_prime + dr) / ed)
            err -= dr
            c += sign_c

        if 2 * err_prime <= dr:
            if r == r1:
                break
            if (dc - err_prime) < ed:
                cc.append(c_prime + sign_c)
                rr.append(r)
                val.append(fabs(dc - err_prime) / ed)
            err += dc
            r += sign_r

    return (np.array(rr, dtype=np.intp),
            np.array(cc, dtype=np.intp),
            1. - np.array(val, dtype=float))
