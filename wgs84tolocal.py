from typing import *
import numpy as np

RADIUS_KM = 6_378.137


def lla_to_wgs84(
        lat: Union[np.ndarray, float],
        lon: Union[np.ndarray, float],
        alt: Union[np.ndarray, float]=0.0,
        round_earth: bool=True
) -> np.ndarray:
    """
    Converts latitude, longitude and altitude (optional) to WGS84 (i.e. XYZ coordinates with origin at the center of
    mass of the earth, the Z axis point to the North Pole, and the X and Y axes pointing to the equator).
    Numpy arrays are also supported.

    :param lat: np.ndarray or float
        The latitude/s.
    :param lon: np.ndarray or float
        The longitude/s.
    :param alt: np.ndarray or float, optional (default: 0, meaning lying on the surface of the earth)
        The altitude/s.
    :param round_earth: bool, optional (default: True)
        If True, earth will be modeled as a sphere; otherwise, it will be modeled as a geoid, as per WGS84
        specification.

    :return: tuple of three np.ndarray or three floats
        The X/s, Y/s and Z/s of the input point/s
    """
    if round_earth:
        FLATTENING = 0
    else:
        FLATTENING = 1 / 298.257_223_563

    cos_lat = np.cos(np.deg2rad(lat))
    sin_lat = np.sin(np.deg2rad(lat))
    cos_lon = np.cos(np.deg2rad(lon))
    sin_lon = np.sin(np.deg2rad(lon))

    c = 1 / np.sqrt(cos_lat * cos_lat + (1 - FLATTENING) * (1 - FLATTENING) * sin_lat * sin_lat)
    s = (1 - FLATTENING) * (1 - FLATTENING) * c
    x = (RADIUS_KM * c + alt) * cos_lat * cos_lon
    y = (RADIUS_KM * c + alt) * cos_lat * sin_lon
    z = (RADIUS_KM * s + alt) * sin_lat

    return np.array([x, y, z])


def get_lla_to_xy_matrix(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        round_earth: bool = True
) -> Tuple[np.array, np.array]:
    """
    Given the coordinates of two points P1 and P2 on the planet, generates two affine transformation matrices to
    convert between WGS84 coordinates and coordinates of a particular system such that:
     - its origin is below the middle point between P1 and P2;
     - the Z axis points outwards from the center of the earth;
     - P1 and P2 are where the Z=0 plane intersects the surface of the earth;
     - P1 + P2 = 0 in the new coordinate system;
     - one unit corresponds to one kilometer.

    :param p1: tuple of two floats
        Latitude and longitude of P1.
    :param p2: tuple of two floats
        Latitude and longitude of P2.
    :param round_earth: bool
        A flag indicating whether earth should be approximated to a sphere; if False, earth is approximated to an oval.
    :return T: numpy array
        A transformation matrix to convert WGS84 coordinates to the new system
    :return T_inv: numpy array
        A transformation matrix to convert coordinates in the new system into WGS84 coordinates
    """
    min_lat = min(p1[0], p2[0])
    max_lat = max(p1[0], p2[0])
    min_lon = min(p1[1], p2[1])
    max_lon = max(p1[1], p2[1])
    a = lla_to_wgs84(min_lat, min_lon, 0.0, round_earth)
    b = lla_to_wgs84(min_lat, max_lon, 0.0, round_earth)
    c = lla_to_wgs84(max_lat, max_lon, 0.0, round_earth)
    m = (a + c) / 2

    # Base vector pointing outwards
    e_3 = m / np.linalg.norm(m)
    # Base vector parallel to the line connecting city1 to city2
    e_1 = (b - a) / np.linalg.norm(b - a)
    # Base vector perpendicular to the other two
    e_2 = np.cross(e_3, e_1)
    e_2 /= np.linalg.norm(e_2)
    e_1 = np.cross(e_2, e_3)

    R = np.column_stack([e_1, e_2, e_3]) # Rotation matrix, 3x3
    z_offset = (R.T @ c)[2]
    p = np.array([0, 0, z_offset]).reshape(-1, 1) # Translation vector, 3x1
    l = np.array([0, 0, 0, 1]).reshape(1, -1) # Last row (0,0,0,1), 1x4
    T = np.block([[R.T, -p], [l]])
    T_inv = np.block([[R, R @ p], [l]])

    return T, T_inv


def lla_to_xy(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        lat: np.ndarray,
        lon: np.ndarray,
        round_earth: bool = True
) -> np.array:
    """
    In the context of planetary coordinates, given the coordinates of two points P1 and P2, converts the coordinates of
    all other given points into a coordinates system such that:
     - its origin is below the middle point between P1 and P2;
     - the Z axis points outwards from the center of the earth;
     - P1 and P2 are where the Z=0 plane intersects the surface of the earth;
     - P1 + P2 = 0 in the new coordinate system;
     - one unit corresponds to one kilometer.

    Only the X and Y coordinates are returned.

    :param p1: tuple of two floats
        Latitude and longitude of P1.
    :param p2: tuple of two floats
        Latitude and longitude of P2.
    :param lat: 1D numpy array (shape n_points)
        The latitude of the points
    :param lon: 1D numpy array (shape n_points)
        The longitude of the points
    :param round_earth: bool
        A flag indicating whether earth should be approximated to a sphere; if False, earth is approximated to an oval.

    :return new_coords: numpy array (shape n_points x 2)
        The X and Y coordinates in the new coordinates system
    """
    T, _ = get_lla_to_xy_matrix(p1, p2, round_earth)
    p = lla_to_wgs84(np.deg2rad(lat), np.deg2rad(lon), 0.0, round_earth)
    p = np.append(p, np.ones((1, p.shape[1])), 0).T # add column of ones
    new_coords = T @ p
    return new_coords[:,:2]


def project_down(p, T):
    d = np.linalg.norm(T[(0, 1, 2), (3, 3, 3)])  # Distance from the origin to the plane
    n = T[(0, 1, 2), (2, 2, 2)] / np.linalg.norm(T[(0, 1, 2), (2, 2, 2)])  # Normal of the plane (a.k.a. e_3)
    pj = p - (np.linalg.norm(n @ p) - d) * n
    return pj



def haversine_distance_km(
        lat1: Union[float, np.ndarray],
        lon1: Union[float, np.ndarray],
        lat2: Union[float, np.ndarray],
        lon2: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculates distance using Haversine formula. The value is returned in kilometers.
    Credits: https://stackoverflow.com/a/44682708/1326534

    :param lat1: np.ndarray
        Latitude of first set of data
    :param lon1: np.ndarray
        Longitude of first set of data
    :param lat2: np.ndarray
        Latitude of first set of data
    :param lon2: np.ndarray
        Longitude of first set of data
    :return: np.ndarray
        Distance calculated with the Haversine formula
    """
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore')

        # Get array data; convert to radians to simulate 'map(radians,...)' part
        lat1 = np.deg2rad(lat1)
        lon1 = np.deg2rad(lon1)
        lat2 = np.deg2rad(lat2)
        lon2 = np.deg2rad(lon2)

        # Get the differentiations
        lat = lat1 - lat2
        lon = lon1 - lon2

        # Compute "sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lng * 0.5) ** 2"
        d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lon * 0.5) ** 2
        h = 2 * RADIUS_KM * np.arcsin(np.sqrt(d))
    return h

