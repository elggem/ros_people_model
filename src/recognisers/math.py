import cv2
import dlib
import numpy as np
from geometry_msgs.msg import Point


def face_landmarks_to_geometry_msgs_points(face_landmarks):
    return [Point(p.x, p.y, 0) for p in face_landmarks.parts()]


def geometry_msgs_points_to_face_landmarks(geometry_msgs_points):
    points = dlib.points()
    [points.append(dlib.point(int(p.x), int(p.y))) for p in geometry_msgs_points]
    return points


def sanitize(image, rescale_size):
    """
        Converts image into gray scale if it RGB image and resize it to IMG_SIZE

        Parameters
        ----------
        image : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            gray scale image resized to IMG_SIZE
        """
    if image is None:
        return None
    assert len(image.shape) == 2 or len(image.shape) == 3, "Image dim should be either 2 or 3. It is " + str(
        len(image.shape))

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, rescale_size)
    return image


def get_distances_angles(all_dlib_points, centroids):
    """
    Get the distances for each dlib facial key points in face from centroid of the points and
    angles between the dlib points vector and centroid vector.

    Parameters
    ----------
    all_dlib_points : numpy.ndarray
        dlib facial key points for each face.
    centroid :
        centroid of dlib facial key point for each face
    Returns
    -------
    numpy.ndarray , numpy.ndarray
        Dlib landmarks distances and angles with respect to respective centroid.
    """
    all_distances = np.zeros((len(all_dlib_points), 1, 68, 1))
    all_angles = np.zeros((len(all_dlib_points), 1, 68, 1))
    for i in range(len(all_dlib_points)):
        dists = np.linalg.norm(centroids[i] - all_dlib_points[i][0], axis=1)
        angles = get_angles(all_dlib_points[i][0], centroids[i])
        all_distances[i][0] = dists.reshape(1, 68, 1)
        all_angles[i][0] = angles.reshape(1, 68, 1)
    return all_distances, all_angles


def angle_between(p1, p2):
    """
    Get clockwise angle between two vectors

    Parameters
    ----------
    p1 : numpy.ndarray
        first vector.
    p2 : numpy.ndarray
        second vector.
    Returns
    -------
    float
        angle in radiuns
    """
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return (ang1 - ang2) % (2 * np.pi)


def angles_between_point_vectors(v1, v2):
    """Calculates angle between two point vectors.
    Parameters
    ----------
    v1 : numpy.ndarray
        First vector
    v2 : numpy.ndarray
        Second vector

    Returns:
    --------
    numpy.ndarray
        Vector if one of the arguments is matrix and scalar if both arguments are vectors.
    """
    dot_prod = (v1 * v2).sum(axis=1)
    v1_norm = np.linalg.norm(v1, axis=1)
    v2_norm = np.linalg.norm(v2, axis=1)

    cosine_of_angle = (dot_prod / (v1_norm * v2_norm)).reshape(11, 1)

    angles = np.arccos(np.clip(cosine_of_angle, -1, 1))

    return angles


def get_angles(dlib_points, centroid):
    """
    Get clockwise angles between dlib landmarks of face and centroid of landmarks.

    Parameters
    ----------
    dlib_points : numpy.ndarray
        dlib landmarks of face.
    centroid : numpy.ndarray
        centroid of dlib landrmask.
    Returns
    -------
    numpy.ndarray
        dlib points clockwise angles in radiuns with respect to centroid vector
    """
    output = np.zeros((68))
    for i in range(68):
        angle = angle_between(dlib_points[i], centroid)
        output[i] = angle
    return output


def arg_max(array):
    """
    Get index of maximum element of 1D array

    Parameters
    ----------
    array : list

    Returns
    -------
    int
        index of maximum element of the array
    """
    max_value = array[0]
    max_index = 0
    for i, el in enumerate(array):
        if max_value < el:
            max_value = el
            max_index = i
    return max_index


def distance_between(v1, v2):
    """Calculates euclidean distance between two vectors.
    If one of the arguments is matrix then the output is calculated for each row
    of that matrix.

    Parameters
    ----------
    v1 : numpy.ndarray
        First vector
    v2 : numpy.ndarray
        Second vector

    Returns:
    --------
    numpy.ndarray
        Matrix if one of the arguments is matrix and vector if both arguments are vectors.
    """

    diff = v2 - v1
    diff_squared = np.square(diff)
    dist_squared = diff_squared.sum(axis=1)
    dists = np.sqrt(dist_squared)
    return dists
