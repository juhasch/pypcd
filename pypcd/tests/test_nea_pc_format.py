import numpy as np
import pytest

try:
    from sensor_msgs.msg import PointField
    HAS_SENSOR_MSGS = True
except ImportError:
    HAS_SENSOR_MSGS = False

from pypcd.nea_pc_format import (
    datatype_to_size,
    make_nea_fields_dicts,
    make_nea_float_fields_dicts,
    field_dict_list_to_dtypes,
    make_nea_dtypes,
    make_nea_float_dtypes,
    field_dict_list_to_pcd_metadata,
)

# Define PointField constants in case sensor_msgs is not available
if not HAS_SENSOR_MSGS:
    class PointField:
        INT8 = 1
        UINT8 = 2
        INT16 = 3
        UINT16 = 4
        INT32 = 5
        UINT32 = 6
        FLOAT32 = 7
        FLOAT64 = 8

@pytest.mark.skipif(not HAS_SENSOR_MSGS, reason="sensor_msgs package not available")
def test_datatype_to_size():
    """Test the datatype_to_size function for all supported datatypes."""
    assert datatype_to_size(PointField.INT8) == 1
    assert datatype_to_size(PointField.UINT8) == 1
    assert datatype_to_size(PointField.INT16) == 2
    assert datatype_to_size(PointField.UINT16) == 2
    assert datatype_to_size(PointField.INT32) == 4
    assert datatype_to_size(PointField.UINT32) == 4
    assert datatype_to_size(PointField.FLOAT32) == 4
    assert datatype_to_size(PointField.FLOAT64) == 8

@pytest.mark.skipif(not HAS_SENSOR_MSGS, reason="sensor_msgs package not available")
def test_make_nea_fields_dicts():
    """Test the make_nea_fields_dicts function with different combinations."""
    # Test without label and padding
    fields = make_nea_fields_dicts(with_label=False, with_padding=False)
    assert len(fields) == 14  # Original fields without label and padding
    assert fields[-1]['name'] == 'return_type'
    
    # Test with label but without padding
    fields = make_nea_fields_dicts(with_label=True, with_padding=False)
    assert len(fields) == 15  # Original fields + label
    assert fields[-1]['name'] == 'label'
    
    # Test with label and padding
    fields = make_nea_fields_dicts(with_label=True, with_padding=True)
    assert len(fields) == 16  # Original fields + label + padding
    assert fields[-1]['name'] == '_PAD'
    assert fields[-1]['count'] == 2

@pytest.mark.skipif(not HAS_SENSOR_MSGS, reason="sensor_msgs package not available")
def test_make_nea_float_fields_dicts():
    """Test the make_nea_float_fields_dicts function with different combinations."""
    # Test without label and padding
    fields = make_nea_float_fields_dicts(with_label=False, with_padding=False)
    assert len(fields) == 14  # Original fields without label and padding
    assert fields[-1]['name'] == 'return_type'
    
    # Test with label but without padding
    fields = make_nea_float_fields_dicts(with_label=True, with_padding=False)
    assert len(fields) == 15  # Original fields + label
    assert fields[-1]['name'] == 'label'
    
    # Test with label and padding
    fields = make_nea_float_fields_dicts(with_label=True, with_padding=True)
    assert len(fields) == 16  # Original fields + label + padding
    assert fields[-1]['name'] == '_PAD'
    assert fields[-1]['count'] == 9

@pytest.mark.skipif(not HAS_SENSOR_MSGS, reason="sensor_msgs package not available")
def test_field_dict_list_to_dtypes():
    """Test the field_dict_list_to_dtypes function."""
    # Create a simple test field list
    test_fields = [
        {'name': 'x', 'datatype': PointField.FLOAT32, 'count': 1},
        {'name': 'rgb', 'datatype': PointField.UINT32, 'count': 1},
    ]
    
    dtypes = field_dict_list_to_dtypes(test_fields)
    assert len(dtypes) == 2
    assert dtypes[0] == ('x', np.dtype('float32'))
    assert dtypes[1] == ('rgb', np.dtype('uint32'))

@pytest.mark.skipif(not HAS_SENSOR_MSGS, reason="sensor_msgs package not available")
def test_make_nea_dtypes():
    """Test the make_nea_dtypes function."""
    # Test without label and padding
    dtypes = make_nea_dtypes(with_label=False, with_padding=False)
    assert len(dtypes) == 14  # Original fields without label and padding
    assert dtypes[0] == ('x', np.dtype('float64'))
    assert dtypes[-1] == ('return_type', np.dtype('uint8'))
    
    # Test with label and padding
    dtypes = make_nea_dtypes(with_label=True, with_padding=True)
    assert len(dtypes) == 16  # Original fields + label + padding
    assert dtypes[-2] == ('label', np.dtype('uint8'))
    assert dtypes[-1] == ('_PAD', np.dtype('uint8'))

@pytest.mark.skipif(not HAS_SENSOR_MSGS, reason="sensor_msgs package not available")
def test_make_nea_float_dtypes():
    """Test the make_nea_float_dtypes function."""
    # Test without label and padding
    dtypes = make_nea_float_dtypes(with_label=False, with_padding=False)
    assert len(dtypes) == 14  # Original fields without label and padding
    assert dtypes[0] == ('x', np.dtype('float32'))
    assert dtypes[-1] == ('return_type', np.dtype('uint8'))
    
    # Test with label and padding
    dtypes = make_nea_float_dtypes(with_label=True, with_padding=True)
    assert len(dtypes) == 16  # Original fields + label + padding
    assert dtypes[-2] == ('label', np.dtype('uint8'))
    assert dtypes[-1] == ('_PAD', np.dtype('uint8'))

@pytest.mark.skipif(not HAS_SENSOR_MSGS, reason="sensor_msgs package not available")
def test_field_dict_list_to_pcd_metadata():
    """Test the field_dict_list_to_pcd_metadata function."""
    test_fields = [
        {'name': 'x', 'datatype': PointField.FLOAT32, 'count': 1},
        {'name': 'y', 'datatype': PointField.FLOAT32, 'count': 1},
        {'name': 'z', 'datatype': PointField.FLOAT32, 'count': 1},
    ]
    
    metadata = field_dict_list_to_pcd_metadata(test_fields)
    
    assert metadata['version'] == 0.7
    assert metadata['fields'] == ['x', 'y', 'z']
    assert metadata['size'] == [4, 4, 4]  # FLOAT32 size is 4
    assert metadata['type'] == ['F', 'F', 'F']  # FLOAT32 type is 'F'
    assert metadata['count'] == [1, 1, 1]
    assert metadata['width'] == 0
    assert metadata['height'] == 1
    assert metadata['viewpoint'] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert metadata['points'] == 0
    assert metadata['data'] == 'ASCII' 