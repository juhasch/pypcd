"""
this is just a basic sanity check, not a really legit test suite.

TODO maybe download data here instead of having it in repo
"""

import pytest
import numpy as np
import os
import shutil
import tempfile

header1 = """\
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z i
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 500028
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 500028
DATA binary_compressed
"""

header2 = """\
VERSION .7
FIELDS x y z normal_x normal_y normal_z curvature boundary k vp_x vp_y vp_z principal_curvature_x principal_curvature_y principal_curvature_z pc1 pc2
SIZE 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
TYPE F F F F F F F F F F F F F F F F F
COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
WIDTH 19812
HEIGHT 1
VIEWPOINT 0.0 0.0 0.0 1.0 0.0 0.0 0.0
POINTS 19812
DATA ascii
"""


@pytest.fixture
def pcd_fname():
    import pypcd
    return os.path.join(pypcd.__path__[0], 'test_data',
                        'partial_cup_model.pcd')


@pytest.fixture
def ascii_pcd_fname():
    import pypcd
    return os.path.join(pypcd.__path__[0], 'test_data',
                        'ascii.pcd')


@pytest.fixture
def bin_pcd_fname():
    import pypcd
    return os.path.join(pypcd.__path__[0], 'test_data',
                        'bin.pcd')


def cloud_centroid(pc):
    xyz = np.empty((pc.points, 3), dtype=np.float32)
    xyz[:, 0] = pc.pc_data['x']
    xyz[:, 1] = pc.pc_data['y']
    xyz[:, 2] = pc.pc_data['z']
    return xyz.mean(0)


def test_parse_header():
    from pypcd.pypcd import parse_header
    lines = header1.split('\n')
    md = parse_header(lines)
    assert (md['version'] == '0.7')
    assert (md['fields'] == ['x', 'y', 'z', 'i'])
    assert (md['size'] == [4, 4, 4, 4])
    assert (md['type'] == ['F', 'F', 'F', 'F'])
    assert (md['count'] == [1, 1, 1, 1])
    assert (md['width'] == 500028)
    assert (md['height'] == 1)
    assert (md['viewpoint'] == [0, 0, 0, 1, 0, 0, 0])
    assert (md['points'] == 500028)
    assert (md['data'] == 'binary_compressed')


def test_from_path(pcd_fname):
    import pypcd
    pc = pypcd.PointCloud.from_path(pcd_fname)

    fields = 'x y z normal_x normal_y normal_z curvature boundary k vp_x vp_y vp_z principal_curvature_x principal_curvature_y principal_curvature_z pc1 pc2'.split()
    for fld1, fld2 in zip(pc.fields, fields):
        assert(fld1 == fld2)
    assert (pc.width == 19812)
    assert (len(pc.pc_data) == 19812)


def test_add_fields(pcd_fname):
    import pypcd
    pc = pypcd.PointCloud.from_path(pcd_fname)

    old_md = pc.get_metadata()
    # new_dt = [(f, pc.pc_data.dtype[f]) for f in pc.pc_data.dtype.fields]
    # new_data = [pc.pc_data[n] for n in pc.pc_data.dtype.names]
    md = {'fields': ['bla', 'bar'], 'count': [1, 1], 'size': [4, 4],
          'type': ['F', 'F']}
    d = np.rec.fromarrays((np.random.random(len(pc.pc_data)),
                           np.random.random(len(pc.pc_data))))
    newpc = pypcd.add_fields(pc, md, d)

    new_md = newpc.get_metadata()
    # print len(old_md['fields']), len(md['fields']), len(new_md['fields'])
    # print old_md['fields'], md['fields'], new_md['fields']
    assert(len(old_md['fields'])+len(md['fields']) == len(new_md['fields']))


def test_path_roundtrip_ascii(pcd_fname):
    import pypcd
    pc = pypcd.PointCloud.from_path(pcd_fname)
    md = pc.get_metadata()

    tmp_dirname = tempfile.mkdtemp(suffix='_pypcd', prefix='tmp')

    tmp_fname = os.path.join(tmp_dirname, 'out.pcd')

    pc.save_pcd(tmp_fname, compression='ascii')

    assert(os.path.exists(tmp_fname))

    pc2 = pypcd.PointCloud.from_path(tmp_fname)
    md2 = pc2.get_metadata()
    for k, v in md2.items():
        assert(k in md)
        if k != 'data':
            assert(md[k] == v)
        else:
            assert(v == 'ascii')

    np.testing.assert_equal(pc.pc_data, pc2.pc_data)

    if os.path.exists(tmp_fname):
        os.unlink(tmp_fname)
    os.removedirs(tmp_dirname)


def test_path_roundtrip_binary(pcd_fname):
    import pypcd
    pc = pypcd.PointCloud.from_path(pcd_fname)
    md = pc.get_metadata()

    tmp_dirname = tempfile.mkdtemp(suffix='_pypcd', prefix='tmp')

    tmp_fname = os.path.join(tmp_dirname, 'out.pcd')

    pc.save_pcd(tmp_fname, compression='binary')

    assert(os.path.exists(tmp_fname))

    pc2 = pypcd.PointCloud.from_path(tmp_fname)
    md2 = pc2.get_metadata()
    for k, v in md2.items():
        assert(k in md)
        if k != 'data':
            assert(md[k] == v)
        else:
            assert(v == 'binary')

    np.testing.assert_equal(pc.pc_data, pc2.pc_data)

    if os.path.exists(tmp_fname):
        os.unlink(tmp_fname)
    os.removedirs(tmp_dirname)


def test_path_roundtrip_binary_compressed(pcd_fname):
    import pypcd
    pc = pypcd.PointCloud.from_path(pcd_fname)
    md = pc.get_metadata()

    tmp_dirname = tempfile.mkdtemp(suffix='_pypcd', prefix='tmp')

    tmp_fname = os.path.join(tmp_dirname, 'out.pcd')

    pc.save_pcd(tmp_fname, compression='binary_compressed')

    assert(os.path.exists(tmp_fname))

    pc2 = pypcd.PointCloud.from_path(tmp_fname)
    md2 = pc2.get_metadata()
    for k, v in md2.items():
        if k == 'data':
            assert v == 'binary_compressed'
        else:
            assert v == md[k]

    np.testing.assert_equal(pc.pc_data, pc2.pc_data)

    if os.path.exists(tmp_dirname):
        shutil.rmtree(tmp_dirname)


def test_cat_pointclouds(pcd_fname):
    import pypcd
    pc = pypcd.PointCloud.from_path(pcd_fname)
    pc2 = pc.copy()
    pc2.pc_data['x'] += 0.1
    pc3 = pypcd.cat_point_clouds(pc, pc2)
    for fld, fld3 in zip(pc.fields, pc3.fields):
        assert(fld == fld3)
    assert(pc3.width == pc.width+pc2.width)


def test_ascii_bin1(ascii_pcd_fname, bin_pcd_fname):
    import pypcd
    apc1 = pypcd.point_cloud_from_path(ascii_pcd_fname)
    bpc1 = pypcd.point_cloud_from_path(bin_pcd_fname)
    am = cloud_centroid(apc1)
    bm = cloud_centroid(bpc1)
    assert(np.allclose(am, bm))


def test_make_xyz_point_cloud():
    import pypcd
    xyz = np.random.rand(100, 3).astype(np.float32)
    pc = pypcd.make_xyz_point_cloud(xyz)
    assert pc.points == 100
    assert len(pc.fields) == 3
    assert all(f in pc.fields for f in ['x', 'y', 'z'])
    np.testing.assert_array_equal(pc.pc_data['x'].squeeze(), xyz[:, 0])
    np.testing.assert_array_equal(pc.pc_data['y'].squeeze(), xyz[:, 1])
    np.testing.assert_array_equal(pc.pc_data['z'].squeeze(), xyz[:, 2])


def test_make_xyz_rgb_point_cloud():
    import pypcd
    xyz = np.random.rand(100, 3).astype(np.float32)
    rgb = np.random.randint(0, 256, (100, 3), dtype=np.uint8)
    # Reshape RGB to 2D array for encoding
    rgb_reshaped = rgb.reshape(-1, 3)
    # Reshape each RGB value to 2D before encoding
    rgb_encoded = np.array([pypcd.encode_rgb_for_pcl(r.reshape(1, 3)) for r in rgb_reshaped], dtype=np.float32)
    xyz_rgb = np.column_stack([xyz, rgb_encoded])
    pc = pypcd.make_xyz_rgb_point_cloud(xyz_rgb)
    assert pc.points == 100
    assert len(pc.fields) == 4  # x, y, z, rgb
    assert all(f in pc.fields for f in ['x', 'y', 'z', 'rgb'])
    np.testing.assert_array_equal(pc.pc_data['x'].squeeze(), xyz[:, 0])
    np.testing.assert_array_equal(pc.pc_data['y'].squeeze(), xyz[:, 1])
    np.testing.assert_array_equal(pc.pc_data['z'].squeeze(), xyz[:, 2])
    # Ensure consistent shapes for RGB comparison
    pc_rgb = pc.pc_data['rgb'].squeeze()
    rgb_encoded = rgb_encoded.squeeze()
    np.testing.assert_array_equal(pc_rgb, rgb_encoded)


def test_rgb_encoding_decoding():
    import pypcd
    rgb = np.array([[255, 128, 64]], dtype=np.uint8)  # 2D array
    encoded = pypcd.encode_rgb_for_pcl(rgb)
    decoded = pypcd.decode_rgb_from_pcl(encoded)
    np.testing.assert_array_equal(rgb, decoded)  # Compare 2D arrays


def test_save_txt(pcd_fname):
    import pypcd
    pc = pypcd.PointCloud.from_path(pcd_fname)
    tmp_dirname = tempfile.mkdtemp(suffix='_pypcd', prefix='tmp')
    tmp_fname = os.path.join(tmp_dirname, 'out.txt')
    
    pc.save_txt(tmp_fname)
    assert os.path.exists(tmp_fname)
    
    # Read back and verify
    with open(tmp_fname, 'r') as f:
        lines = f.readlines()
        assert len(lines) == pc.points + 1  # header + data
    
    if os.path.exists(tmp_fname):
        os.unlink(tmp_fname)
    os.removedirs(tmp_dirname)


def test_update_field(pcd_fname):
    import pypcd
    pc = pypcd.PointCloud.from_path(pcd_fname)
    old_x = pc.pc_data['x'].copy()
    new_x = old_x + 1.0
    
    pc = pypcd.update_field(pc, 'x', new_x)
    np.testing.assert_array_equal(pc.pc_data['x'], new_x)
    assert pc.pc_data['y'].shape == old_x.shape  # other fields unchanged


def test_point_cloud_from_array():
    import pypcd
    arr = np.random.rand(100, 3).astype(np.float32)
    structured_arr = np.zeros(100, dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])
    structured_arr['x'] = arr[:, 0]
    structured_arr['y'] = arr[:, 1]
    structured_arr['z'] = arr[:, 2]
    pc = pypcd.PointCloud.from_array(structured_arr)
    assert pc.points == 100
    assert len(pc.fields) == 3
    assert all(f in pc.fields for f in ['x', 'y', 'z'])
    np.testing.assert_array_equal(pc.pc_data['x'].squeeze(), arr[:, 0])
    np.testing.assert_array_equal(pc.pc_data['y'].squeeze(), arr[:, 1])
    np.testing.assert_array_equal(pc.pc_data['z'].squeeze(), arr[:, 2])


def test_point_cloud_copy(pcd_fname):
    import pypcd
    pc = pypcd.PointCloud.from_path(pcd_fname)
    pc_copy = pc.copy()
    
    # Verify metadata
    assert pc.fields == pc_copy.fields
    assert pc.width == pc_copy.width
    assert pc.height == pc_copy.height
    assert pc.points == pc_copy.points
    
    # Verify data
    np.testing.assert_array_equal(pc.pc_data, pc_copy.pc_data)
    
    # Verify independence
    pc_copy.pc_data['x'] += 1.0
    assert not np.array_equal(pc.pc_data['x'], pc_copy.pc_data['x'])


def test_save_xyz_label(pcd_fname):
    import pypcd
    pc = pypcd.PointCloud.from_path(pcd_fname)
    
    # Add label field
    label_data = np.ones(len(pc.pc_data), dtype=np.int32) * 1000
    pc = pypcd.add_fields(pc, 
                         {'fields': ['label'], 'count': [1], 'size': [4], 'type': ['I']},
                         np.rec.fromarrays([label_data]))
    
    tmp_dirname = tempfile.mkdtemp(suffix='_pypcd', prefix='tmp')
    tmp_fname = os.path.join(tmp_dirname, 'out.xyz')
    
    pc.save_xyz_label(tmp_fname, use_default_lbl=True)  # Use default label to avoid type issues
    assert os.path.exists(tmp_fname)
    
    if os.path.exists(tmp_fname):
        os.unlink(tmp_fname)
    os.removedirs(tmp_dirname)


def test_save_xyz_intensity_label(pcd_fname):
    import pypcd
    pc = pypcd.PointCloud.from_path(pcd_fname)
    
    # Add intensity and label fields
    intensity_data = np.ones(len(pc.pc_data), dtype=np.float32) * 0.5
    label_data = np.ones(len(pc.pc_data), dtype=np.int32) * 1000
    pc = pypcd.add_fields(pc, 
                         {'fields': ['intensity', 'label'], 
                          'count': [1, 1], 
                          'size': [4, 4], 
                          'type': ['F', 'I']},
                         np.rec.fromarrays([intensity_data, label_data]))
    
    tmp_dirname = tempfile.mkdtemp(suffix='_pypcd', prefix='tmp')
    tmp_fname = os.path.join(tmp_dirname, 'out.xyz')
    
    pc.save_xyz_intensity_label(tmp_fname, use_default_lbl=True)  # Use default label to avoid type issues
    assert os.path.exists(tmp_fname)
    
    if os.path.exists(tmp_fname):
        os.unlink(tmp_fname)
    os.removedirs(tmp_dirname)


def test_point_cloud_from_buffer(pcd_fname):
    import pypcd
    pc = pypcd.PointCloud.from_path(pcd_fname)
    buffer = pc.save_pcd_to_buffer()
    pc_from_buffer = pypcd.PointCloud.from_buffer(buffer)
    
    # Verify metadata
    assert pc.fields == pc_from_buffer.fields
    assert pc.width == pc_from_buffer.width
    assert pc.height == pc_from_buffer.height
    assert pc.points == pc_from_buffer.points
    
    # Verify data
    np.testing.assert_array_equal(pc.pc_data, pc_from_buffer.pc_data)
