import numpy as np
import pytest
from pypcd.sautil import (
    transform_xyz,
    transform_cloud_array,
    flip_around_x,
    get_xyz_array,
    get_xyz_viewpoint_array,
    get_xyzl_array,
)


def test_transform_xyz():
    # Test identity transformation
    T_identity = np.eye(4)
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = transform_xyz(T_identity, xyz)
    np.testing.assert_array_almost_equal(result, xyz)

    # Test translation
    T_translation = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 2],
        [0, 0, 1, 3],
        [0, 0, 0, 1]
    ])
    expected = xyz + np.array([1, 2, 3])
    result = transform_xyz(T_translation, xyz)
    np.testing.assert_array_almost_equal(result, expected)


def test_transform_cloud_array():
    # Create a structured array with xyz coordinates
    pc_data = np.zeros(2, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    pc_data['x'] = [1.0, 4.0]
    pc_data['y'] = [2.0, 5.0]
    pc_data['z'] = [3.0, 6.0]

    # Test identity transformation
    T_identity = np.eye(4)
    result = transform_cloud_array(T_identity, pc_data)
    np.testing.assert_array_almost_equal(result['x'], pc_data['x'])
    np.testing.assert_array_almost_equal(result['y'], pc_data['y'])
    np.testing.assert_array_almost_equal(result['z'], pc_data['z'])

    # Test with origin points
    pc_data_with_origin = np.zeros(2, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('x_origin', 'f4'), ('y_origin', 'f4'), ('z_origin', 'f4')
    ])
    pc_data_with_origin['x'] = [1.0, 4.0]
    pc_data_with_origin['y'] = [2.0, 5.0]
    pc_data_with_origin['z'] = [3.0, 6.0]
    pc_data_with_origin['x_origin'] = [0.0, 0.0]
    pc_data_with_origin['y_origin'] = [0.0, 0.0]
    pc_data_with_origin['z_origin'] = [0.0, 0.0]

    result = transform_cloud_array(T_identity, pc_data_with_origin)
    np.testing.assert_array_almost_equal(result['x_origin'], pc_data_with_origin['x_origin'])
    np.testing.assert_array_almost_equal(result['y_origin'], pc_data_with_origin['y_origin'])
    np.testing.assert_array_almost_equal(result['z_origin'], pc_data_with_origin['z_origin'])


def test_flip_around_x():
    # Create a structured array with xyz coordinates
    pc_data = np.zeros(2, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    pc_data['x'] = [1.0, 4.0]
    pc_data['y'] = [2.0, 5.0]
    pc_data['z'] = [3.0, 6.0]

    # Test flipping
    flip_around_x(pc_data)
    np.testing.assert_array_almost_equal(pc_data['x'], [1.0, 4.0])  # x should remain unchanged
    np.testing.assert_array_almost_equal(pc_data['y'], [-2.0, -5.0])  # y should be negated
    np.testing.assert_array_almost_equal(pc_data['z'], [-3.0, -6.0])  # z should be negated

    # Test with origin points
    pc_data_with_origin = np.zeros(2, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('x_origin', 'f4'), ('y_origin', 'f4'), ('z_origin', 'f4')
    ])
    pc_data_with_origin['x'] = [1.0, 4.0]
    pc_data_with_origin['y'] = [2.0, 5.0]
    pc_data_with_origin['z'] = [3.0, 6.0]
    pc_data_with_origin['x_origin'] = [0.0, 0.0]
    pc_data_with_origin['y_origin'] = [1.0, 2.0]
    pc_data_with_origin['z_origin'] = [1.0, 2.0]

    flip_around_x(pc_data_with_origin)
    np.testing.assert_array_almost_equal(pc_data_with_origin['x_origin'], [0.0, 0.0])  # x_origin should remain unchanged
    np.testing.assert_array_almost_equal(pc_data_with_origin['y_origin'], [-1.0, -2.0])  # y_origin should be negated
    np.testing.assert_array_almost_equal(pc_data_with_origin['z_origin'], [-1.0, -2.0])  # z_origin should be negated


def test_get_xyz_array():
    # Create a structured array with xyz coordinates
    pc_data = np.zeros(2, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    pc_data['x'] = [1.0, 4.0]
    pc_data['y'] = [2.0, 5.0]
    pc_data['z'] = [3.0, 6.0]

    # Test getting xyz array
    xyz = get_xyz_array(pc_data)
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_almost_equal(xyz, expected)

    # Test with different dtype
    xyz = get_xyz_array(pc_data, dtype=np.float64)
    assert xyz.dtype == np.float64


def test_get_xyz_viewpoint_array():
    # Create a structured array with xyz and origin coordinates
    pc_data = np.zeros(2, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('x_origin', 'f4'), ('y_origin', 'f4'), ('z_origin', 'f4')
    ])
    pc_data['x_origin'] = [1.0, 4.0]
    pc_data['y_origin'] = [2.0, 5.0]
    pc_data['z_origin'] = [3.0, 6.0]

    # Test getting xyz viewpoint array
    xyz = get_xyz_viewpoint_array(pc_data)
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_almost_equal(xyz, expected)

    # Test with different dtype
    xyz = get_xyz_viewpoint_array(pc_data, dtype=np.float64)
    assert xyz.dtype == np.float64


def test_get_xyzl_array():
    # Create a structured array with xyz and label coordinates
    pc_data = np.zeros(2, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'f4')])
    pc_data['x'] = [1.0, 4.0]
    pc_data['y'] = [2.0, 5.0]
    pc_data['z'] = [3.0, 6.0]
    pc_data['label'] = [0.0, 1.0]

    # Test getting xyzl array
    xyzl = get_xyzl_array(pc_data)
    expected = np.array([[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 1.0]])
    np.testing.assert_array_almost_equal(xyzl, expected)

    # Test with different dtype
    xyzl = get_xyzl_array(pc_data, dtype=np.float64)
    assert xyzl.dtype == np.float64 