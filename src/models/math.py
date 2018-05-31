import numpy as np
from geometry_msgs.msg import Point


def position_is_close(p1, p2, close=0.5):
    return np.linalg.norm([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z]) < close


def blend_positions(p1, p2, bld_pos=0.35, bld_z=0.95):
    pt = Point()
    pt.x = (p1.x * (1.0 - bld_pos)) + (p2.x * bld_pos)
    pt.y = (p1.y * (1.0 - bld_pos)) + (p2.y * bld_pos)
    pt.z = (p1.z * (1.0 - bld_z)) + (p2.z * bld_z)
    return pt


def position_of_feature(feature):
    pt = Point()
    pt.x = feature.roi.x_offset + (feature.roi.width / 2)
    pt.y = feature.roi.y_offset + (feature.roi.height / 2)
    pt.z = feature.roi.width * feature.roi.height * 0.00095
    return pt
