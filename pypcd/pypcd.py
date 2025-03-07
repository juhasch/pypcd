"""
Read and write PCL .pcd files in python.

This module provides functionality to work with Point Cloud Data (PCD) files,
which are commonly used in the Point Cloud Library (PCL). It supports reading
and writing PCD files in various formats (ASCII, binary, binary_compressed),
as well as converting between different point cloud representations.

Key features:
- Reading and writing PCD files in ASCII, binary, and binary_compressed formats
- Converting between numpy arrays and point cloud data
- Creating point clouds with XYZ, RGB, and label data
- Concatenating point clouds
- Adding and updating fields in point clouds
- Converting between PCL and ROS point cloud messages (if sensor_msgs is available)

Author: dimatura@cmu.edu, 2013-2018

TODO:
- Better API for wacky operations
- Add a CLI for common operations
- Deal properly with padding
- Deal properly with multicount fields
- Better support for RGB nonsense
"""

import re
import struct
import copy
from io import BytesIO as sio

import numpy as np
import warnings
import lzf

HAS_SENSOR_MSGS = True
try:
    from sensor_msgs.msg import PointField
    import numpy_pc2  # needs sensor_msgs
except ImportError:
    HAS_SENSOR_MSGS = False

__all__ = ['PointCloud',
           'point_cloud_to_path',
           'point_cloud_to_buffer',
           'point_cloud_to_fileobj',
           'point_cloud_from_path',
           'point_cloud_from_buffer',
           'point_cloud_from_fileobj',
           'make_xyz_point_cloud',
           'make_xyz_rgb_point_cloud',
           'make_xyz_label_point_cloud',
           'save_txt',
           'cat_point_clouds',
           'add_fields',
           'update_field',
           'build_ascii_fmtstr',
           'encode_rgb_for_pcl',
           'decode_rgb_from_pcl',
           'save_point_cloud',
           'save_point_cloud_bin',
           'save_point_cloud_bin_compressed',
           'pcd_type_to_numpy_type',
           'numpy_type_to_pcd_type',
           ]

if HAS_SENSOR_MSGS:
    pc2_pcd_type_mappings = [(PointField.INT8, ('I', 1)),
                             (PointField.UINT8, ('U', 1)),
                             (PointField.INT16, ('I', 2)),
                             (PointField.UINT16, ('U', 2)),
                             (PointField.INT32, ('I', 4)),
                             (PointField.UINT32, ('U', 4)),
                             (PointField.FLOAT32, ('F', 4)),
                             (PointField.FLOAT64, ('F', 8))]
    pc2_type_to_pcd_type = dict(pc2_pcd_type_mappings)
    pcd_type_to_pc2_type = dict((q, p) for (p, q) in pc2_pcd_type_mappings)
    __all__.extend(['pcd_type_to_pc2_type', 'pc2_type_to_pcd_type'])

numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


def parse_header(lines):
    """ Parse header of PCD files.
    
    Extracts metadata from the header lines of a PCD file.
    
    Args:
        lines: List of strings containing the header lines of a PCD file
        
    Returns:
        Dictionary containing the parsed metadata
    """
    metadata = {}
    for ln in lines:
        if ln.startswith('#') or len(ln) < 2:
            continue
        match = re.match(r'(\w+)\s+([\w\s\.]+)', ln)
        if not match:
            warnings.warn("warning: can't understand line: %s" % ln)
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = list(map(int, value.split()))
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = list(map(float, value.split()))
        elif key == 'data':
            metadata[key] = value.strip().lower()
        # TODO apparently count is not required?
    # add some reasonable defaults
    if 'count' not in metadata:
        metadata['count'] = [1]*len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata


def write_header(metadata, rename_padding=False):
    """ Given metadata as dictionary, return a string header.
    
    Creates a PCD header string from metadata.
    
    Args:
        metadata: Dictionary containing PCD metadata
        rename_padding: If True, rename '_' fields to 'padding'
        
    Returns:
        String containing the PCD header
    """
    template = """\
VERSION {version}
FIELDS {fields}
SIZE {size}
TYPE {type}
COUNT {count}
WIDTH {width}
HEIGHT {height}
VIEWPOINT {viewpoint}
POINTS {points}
DATA {data}
"""
    str_metadata = metadata.copy()

    if not rename_padding:
        str_metadata['fields'] = ' '.join(metadata['fields'])
    else:
        new_fields = []
        for f in metadata['fields']:
            if f == '_':
                new_fields.append('padding')
            else:
                new_fields.append(f)
        str_metadata['fields'] = ' '.join(new_fields)
    str_metadata['size'] = ' '.join(map(str, metadata['size']))
    str_metadata['type'] = ' '.join(metadata['type'])
    str_metadata['count'] = ' '.join(map(str, metadata['count']))
    str_metadata['width'] = str(metadata['width'])
    str_metadata['height'] = str(metadata['height'])
    str_metadata['viewpoint'] = ' '.join(map(str, metadata['viewpoint']))
    str_metadata['points'] = str(metadata['points'])
    tmpl = template.format(**str_metadata)
    return tmpl


def _metadata_is_consistent(metadata):
    """ Sanity check for metadata. Just some basic checks.
    
    Verifies that the metadata dictionary contains all required fields and that
    the values are consistent with each other.
    
    Args:
        metadata: Dictionary containing PCD metadata
        
    Returns:
        bool: True if the metadata is consistent, False otherwise
    """
    checks = []
    required = ('version', 'fields', 'size', 'width', 'height', 'points',
                'viewpoint', 'data')
    for f in required:
        if f not in metadata:
            print('%s required' % f)
    checks.append((lambda m: all([k in m for k in required]),
                   'missing field'))
    checks.append((lambda m: len(m['type']) == len(m['count']) ==
                   len(m['fields']),
                   'length of type, count and fields must be equal'))
    checks.append((lambda m: m['height'] > 0,
                   'height must be greater than 0'))
    checks.append((lambda m: m['width'] > 0,
                   'width must be greater than 0'))
    checks.append((lambda m: m['points'] > 0,
                   'points must be greater than 0'))
    checks.append((lambda m: m['data'].lower() in ('ascii', 'binary',
                   'binary_compressed'),
                   'unknown data type:'
                   'should be ascii/binary/binary_compressed'))
    ok = True
    for check, msg in checks:
        if not check(metadata):
            print('error:', msg)
            ok = False
    return ok


def _build_dtype(metadata):
    """ Build numpy structured array dtype from pcl metadata.
    
    Creates a numpy structured array dtype based on the field information in the metadata.
    Fields with count > 1 are 'flattened' by creating multiple single-count fields.
    
    Args:
        metadata: Dictionary containing PCD metadata
        
    Returns:
        numpy.dtype: A structured array dtype for the point cloud data
        
    Note:
        Fields with count > 1 are 'flattened' by creating multiple single-count fields.
        For example, a field 'normal' with count 3 becomes 'normal_0000', 'normal_0001', 'normal_0002'.
        
    TODO:
        Allow 'proper' multi-count fields.
    """
    fieldnames = []
    typenames = []
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            fieldnames.extend(['%s_%04d' % (f, i) for i in range(c)])
            typenames.extend([np_type]*c)
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype


def build_ascii_fmtstr(pc):
    """ Make a format string for printing to ascii.
    
    Creates a format string for saving point cloud data in ASCII format.
    
    Args:
        pc: A PointCloud object
        
    Returns:
        List of format strings for each field
        
    Raises:
        ValueError: If a field has an unknown type
        
    Note:
        Uses %.10f for float fields, %d for integer fields, and %u for unsigned integer fields.
    """
    fmtstr = []
    for t, cnt in zip(pc.type, pc.count):
        if t == 'F':
            fmtstr.extend(['%.10f']*cnt)
        elif t == 'I':
            fmtstr.extend(['%d']*cnt)
        elif t == 'U':
            fmtstr.extend(['%u']*cnt)
        else:
            raise ValueError("don't know about type %s" % t)
    return fmtstr


def parse_ascii_pc_data(f, dtype, metadata):
    """ Use numpy to parse ascii pointcloud data.
    
    Parses ASCII point cloud data from a file object.
    
    Args:
        f: File object containing ASCII point cloud data
        dtype: Numpy dtype for the point cloud data
        metadata: Dictionary containing PCD metadata
        
    Returns:
        Numpy structured array containing the point cloud data
    """
    return np.loadtxt(f, dtype=dtype, delimiter=' ')


def parse_binary_pc_data(f, dtype, metadata):
    """ Parse binary pointcloud data.
    
    Parses binary point cloud data from a file object.
    
    Args:
        f: File object containing binary point cloud data
        dtype: Numpy dtype for the point cloud data
        metadata: Dictionary containing PCD metadata
        
    Returns:
        Numpy structured array containing the point cloud data
    """
    rowstep = metadata['points']*dtype.itemsize
    # for some reason pcl adds empty space at the end of files
    buf = f.read(rowstep)
    return np.frombuffer(buf, dtype=dtype)


def parse_binary_compressed_pc_data(f, dtype, metadata):
    """ Parse lzf-compressed data.
    
    Parses binary compressed point cloud data from a file object.
    The format is undocumented but seems to be:
    - compressed size of data (uint32)
    - uncompressed size of data (uint32)
    - compressed data
    - junk
    
    Args:
        f: File object containing binary compressed point cloud data
        dtype: Numpy dtype for the point cloud data
        metadata: Dictionary containing PCD metadata
        
    Returns:
        Numpy structured array containing the point cloud data
        
    Raises:
        IOError: If there is an error decompressing the data
    """
    fmt = 'II'
    compressed_size, uncompressed_size =\
        struct.unpack(fmt, f.read(struct.calcsize(fmt)))
    compressed_data = f.read(compressed_size)
    # TODO what to use as second argument? if buf is None
    # (compressed > uncompressed)
    # should we read buf as raw binary?
    buf = lzf.decompress(compressed_data, uncompressed_size)
    if len(buf) != uncompressed_size:
        raise IOError('Error decompressing data')
    # the data is stored field-by-field
    pc_data = np.zeros(metadata['width'], dtype=dtype)
    ix = 0
    for dti in range(len(dtype)):
        dt = dtype[dti]
        bytes = dt.itemsize * metadata['width']
        column = np.frombuffer(buf[ix:(ix+bytes)], dt)
        pc_data[dtype.names[dti]] = column
        ix += bytes
    return pc_data


def point_cloud_from_fileobj(f):
    """ Parse pointcloud coming from file object f
    
    Reads a PCD file from a file object and returns a PointCloud instance.
    This function automatically detects the PCD format (ASCII, binary, or binary_compressed)
    from the header and parses the data accordingly.
    
    Args:
        f: A file object opened in binary mode ('rb')
        
    Returns:
        A PointCloud object containing the point cloud data
    """
    header = []
    while True:
        ln = f.readline().strip()
        if isinstance(ln, bytes):
            ln = ln.decode('ascii')
        header.append(ln)
        if ln.startswith('DATA'):
            metadata = parse_header(header)
            dtype = _build_dtype(metadata)
            break
    if metadata['data'] == 'ascii':
        pc_data = parse_ascii_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary':
        pc_data = parse_binary_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary_compressed':
        pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
    else:
        print('DATA field is neither "ascii" or "binary" or\
                "binary_compressed"')
    return PointCloud(metadata, pc_data)


def point_cloud_from_path(fname):
    """ Load point cloud from a PCD file.
    
    Opens a PCD file and returns a PointCloud instance.
    
    Args:
        fname: Path to the PCD file
        
    Returns:
        A PointCloud object containing the point cloud data
    """
    with open(fname, 'rb') as f:
        pc = point_cloud_from_fileobj(f)
    return pc


def point_cloud_from_buffer(buf):
    """ Load point cloud from a buffer containing PCD data.
    
    Parses PCD data from a buffer (bytes or BytesIO) and returns a PointCloud instance.
    
    Args:
        buf: Buffer containing PCD data
        
    Returns:
        A PointCloud object containing the point cloud data
    """
    fileobj = sio(buf)
    pc = point_cloud_from_fileobj(fileobj)
    fileobj.close()  # necessary?
    return pc


def point_cloud_to_fileobj(pc, fileobj, data_compression=None):
    """Write pointcloud as .pcd to fileobj.

    Args:
        pc: A PointCloud object
        fileobj: A file object opened in appropriate mode ('w' for ASCII, 'wb' for binary)
        data_compression: Optional compression type ('ascii', 'binary', or 'binary_compressed').
        If None, uses the compression type specified in the PointCloud object

    Raises:
        TypeError: If binary data is being written to a text file object
        ValueError: If the data type is unknown
    """
    metadata = pc.get_metadata()
    if data_compression is not None:
        data_compression = data_compression.lower()
        if data_compression not in ('ascii', 'binary', 'binary_compressed'):
            raise ValueError(f"Invalid compression type: {data_compression}. "
                          f"Must be one of 'ascii', 'binary', or 'binary_compressed'")
        metadata['data'] = data_compression

    # Check if we need binary mode before writing anything
    needs_binary = metadata['data'].lower() in ('binary', 'binary_compressed')
    if needs_binary and not isinstance(fileobj, (sio, bytes)):
        if hasattr(fileobj, 'mode') and 'b' not in fileobj.mode:
            raise TypeError("Binary data can only be written to binary file objects")

    header = write_header(metadata)
    # Write header as string or bytes depending on file mode
    if isinstance(fileobj, (sio, bytes)) or (hasattr(fileobj, 'mode') and 'b' in fileobj.mode):
        fileobj.write(header.encode('ascii'))
    else:
        fileobj.write(header)

    if metadata['data'].lower() == 'ascii':
        fmtstr = build_ascii_fmtstr(pc)
        np.savetxt(fileobj, pc.pc_data, fmt=fmtstr)
    elif metadata['data'].lower() == 'binary':
        # Ensure the data is in the correct byte order for binary output
        pc_data = pc.pc_data.copy()
        for field in pc_data.dtype.names:
            pc_data[field] = pc_data[field].astype(pc_data.dtype[field])
        fileobj.write(pc_data.tobytes())
    elif metadata['data'].lower() == 'binary_compressed':
        # Reorder to column-by-column
        uncompressed_lst = []
        for fieldname in pc.pc_data.dtype.names:
            column = np.ascontiguousarray(pc.pc_data[fieldname]).tobytes()
            uncompressed_lst.append(column)
        uncompressed = b''.join(uncompressed_lst)
        uncompressed_size = len(uncompressed)
        buf = lzf.compress(uncompressed)
        if buf is None:
            # compression didn't shrink the file
            buf = uncompressed
            compressed_size = uncompressed_size
        else:
            compressed_size = len(buf)
        fmt = 'II'
        fileobj.write(struct.pack(fmt, compressed_size, uncompressed_size))
        fileobj.write(buf)
    else:
        raise ValueError(f"Unknown DATA type: {metadata['data']}. "
                         f"Must be one of 'ascii', 'binary', or 'binary_compressed'")
    # we can't close because if it's stringio buf then we can't get value after


def point_cloud_to_path(pc, fname):
    """ Write pointcloud to a PCD file.
    
    Saves a PointCloud object to a file in PCD format.
    
    Args:
        pc: A PointCloud object
        fname: Path to the output PCD file
    """
    mode = 'wb' if pc.data.lower() in ('binary', 'binary_compressed') else 'w'
    with open(fname, mode) as f:
        point_cloud_to_fileobj(pc, f)


def point_cloud_to_buffer(pc, data_compression=None):
    """Write pointcloud to a buffer in PCD format.

    Args:
        pc: A PointCloud object
        data_compression: Optional compression type ('ascii', 'binary', or 'binary_compressed').
        If None, uses the compression type specified in the PointCloud object

    Returns:
        A buffer containing the PCD data
    """
    fileobj = sio()
    point_cloud_to_fileobj(pc, fileobj, data_compression)
    return fileobj.getvalue()


def save_point_cloud(pc, fname):
    """ Save pointcloud to fname in ascii format.
    
    A convenience function to save a PointCloud object to a file in ASCII format.
    
    Args:
        pc: A PointCloud object
        fname: Path to the output PCD file
    """
    pc.save_pcd(fname, compression='ascii')


def save_point_cloud_bin(pc, fname):
    """ Save pointcloud to fname in binary format.
    
    A convenience function to save a PointCloud object to a file in binary format.
    
    Args:
        pc: A PointCloud object
        fname: Path to the output PCD file
    """
    pc.save_pcd(fname, compression='binary')


def save_point_cloud_bin_compressed(pc, fname):
    """ Save pointcloud to fname in binary compressed format.
    
    A convenience function to save a PointCloud object to a file in binary compressed format.
    
    Args:
        pc: A PointCloud object
        fname: Path to the output PCD file
    """
    pc.save_pcd(fname, compression='binary_compressed')


def save_xyz_label(pc, fname, use_default_lbl=False):
    """Save a simple (x y z label) pointcloud, ignoring all other features.

    Args:
        pc: A PointCloud object
        fname: Path to the output text file
        use_default_lbl: If True, use a default label of 1000 when the point cloud
                        doesn't have a label field

    Raises:
        Exception: If the point cloud doesn't have a label field and use_default_lbl is False
    """
    md = pc.get_metadata()
    if not use_default_lbl and ('label' not in md['fields']):
        raise Exception('label is not a field in this point cloud')
    with open(fname, 'w') as f:
        for i in range(pc.points):
            x, y, z = ['%.4f' % d for d in (
                pc.pc_data['x'][i], pc.pc_data['y'][i], pc.pc_data['z'][i]
                )]
            lbl = '1000' if use_default_lbl else pc.pc_data['label'][i]
            f.write(' '.join((x, y, z, lbl))+'\n')


def save_xyz_intensity_label(pc, fname, use_default_lbl=False):
    """Save XYZI point cloud with labels.

    Args:
        pc: A PointCloud object
        fname: Path to the output text file
        use_default_lbl: If True, use a default label of 1000 when the point cloud
                        doesn't have a label field

    Raises:
        Exception: If the point cloud doesn't have a label field and use_default_lbl is False
        Exception: If the point cloud doesn't have an intensity field
    """
    md = pc.get_metadata()
    if not use_default_lbl and ('label' not in md['fields']):
        raise Exception('label is not a field in this point cloud')
    if 'intensity' not in md['fields']:
        raise Exception('intensity is not a field in this point cloud')
    with open(fname, 'w') as f:
        for i in range(pc.points):
            x, y, z = ['%.4f' % d for d in (
                pc.pc_data['x'][i], pc.pc_data['y'][i], pc.pc_data['z'][i]
                )]
            intensity = '%.4f' % pc.pc_data['intensity'][i]
            lbl = '1000' if use_default_lbl else pc.pc_data['label'][i]
            f.write(' '.join((x, y, z, intensity, lbl))+'\n')


def save_txt(pc, fname, header=True):
    """ Save to csv-style text file, separated by spaces.
    
    Saves a point cloud to a text file with space-separated values.
    
    Args:
        pc: A PointCloud object
        fname: Path to the output text file
        header: If True, include a header line with field names
        
    TODO:
    - support multi-count fields.
    - other delimiters.
    """
    with open(fname, 'w') as f:
        if header:
            header_lst = []
            for field_name, cnt in zip(pc.fields, pc.count):
                if cnt == 1:
                    header_lst.append(field_name)
                else:
                    for c in range(cnt):
                        header_lst.append('%s_%04d' % (field_name, c))
            f.write(' '.join(header_lst)+'\n')
        fmtstr = build_ascii_fmtstr(pc)
        np.savetxt(f, pc.pc_data, fmt=fmtstr)


def update_field(pc, field, pc_data):
    """ Updates field in-place.
    
    Updates a specific field in a point cloud with new data.
    
    Args:
        pc: A PointCloud object
        field: The name of the field to update
        pc_data: The new data for the field
        
    Returns:
        The updated PointCloud object
    """
    pc.pc_data[field] = pc_data
    return pc


def add_fields(pc, metadata, pc_data):
    """Build a new point cloud with additional fields.

    Args:
        pc: Base PointCloud object
        metadata: Dictionary containing metadata for the new fields
        pc_data: Numpy structured array containing the new field data

    Returns:
        A new PointCloud object with the additional fields

    Note:
        The new fields are appended to the existing ones.
    """
    new_metadata = pc.get_metadata()
    new_metadata['fields'].extend(metadata['fields'])
    new_metadata['count'].extend(metadata['count'])
    new_metadata['size'].extend(metadata['size'])
    new_metadata['type'].extend(metadata['type'])

    # Create dtype for new point cloud
    old_dtype = pc.pc_data.dtype
    new_dtype = [(n, pc.pc_data.dtype[n]) for n in pc.pc_data.dtype.names]
    for n, c, s, t in zip(metadata['fields'], metadata['count'],
                         metadata['size'], metadata['type']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            new_dtype.append((n, np_type))
        else:
            new_dtype.append((n, np_type, c))

    new_data = np.empty(len(pc.pc_data), new_dtype)
    for n in pc.pc_data.dtype.names:
        new_data[n] = pc.pc_data[n]
    for n, n_tmp in zip(metadata['fields'], pc_data.dtype.names):
        new_data[n] = pc_data[n_tmp]

    return PointCloud(new_metadata, new_data)


def cat_point_clouds(pc1, pc2):
    """ Concatenate two point clouds into bigger point cloud.
    
    Combines two point clouds into a single point cloud. The point clouds must have
    the same fields.
    
    Args:
        pc1: First PointCloud object
        pc2: Second PointCloud object
        
    Returns:
        A new PointCloud object containing points from both input point clouds
        
    Raises:
        ValueError: If the point clouds have different fields
    """
    if len(pc1.fields) != len(pc2.fields):
        raise ValueError("Pointclouds must have same fields")
    new_metadata = pc1.get_metadata()
    new_data = np.concatenate((pc1.pc_data, pc2.pc_data))
    # TODO this only makes sense for unstructured pc?
    new_metadata['width'] = pc1.width+pc2.width
    new_metadata['points'] = pc1.points+pc2.points
    pc3 = PointCloud(new_metadata, new_data)
    return pc3


def make_xyz_point_cloud(xyz, metadata=None):
    """ Make a pointcloud object from xyz array.
    
    Creates a point cloud from an array of XYZ coordinates.
    
    Args:
        xyz: A numpy array of shape (N, 3) containing XYZ coordinates
        metadata: Optional dictionary with additional metadata to include
        
    Returns:
        A PointCloud object
        
    Note:
        The xyz array is cast to float32.
    """
    md = {'version': .7,
          'fields': ['x', 'y', 'z'],
          'size': [4, 4, 4],
          'type': ['F', 'F', 'F'],
          'count': [1, 1, 1],
          'width': len(xyz),
          'height': 1,
          'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          'points': len(xyz),
          'data': 'binary'}
    if metadata is not None:
        md.update(metadata)
    xyz = xyz.astype(np.float32)
    pc_data = xyz.view(np.dtype([('x', np.float32),
                                 ('y', np.float32),
                                 ('z', np.float32)]))
    # pc_data = np.rec.fromarrays([xyz[:,0], xyz[:,1], xyz[:,2]], dtype=dt)
    # data = np.rec.fromarrays([xyz.T], dtype=dt)
    pc = PointCloud(md, pc_data)
    return pc


def make_xyz_rgb_point_cloud(xyz_rgb, metadata=None):
    """Make a pointcloud object from xyz and rgb data.

    Args:
        xyz_rgb: A numpy array of shape (N, 4) containing XYZ coordinates and RGB values.
        The RGB values should be encoded as a single float32 according to PCL conventions
        metadata: Optional dictionary with additional metadata to include

    Returns:
        A PointCloud object

    Raises:
        ValueError: If the input array is not of type float32

    Note:
        The RGB values should be encoded using encode_rgb_for_pcl()
    """
    md = {'version': .7,
          'fields': ['x', 'y', 'z', 'rgb'],
          'count': [1, 1, 1, 1],
          'width': len(xyz_rgb),
          'height': 1,
          'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          'points': len(xyz_rgb),
          'type': ['F', 'F', 'F', 'F'],
          'size': [4, 4, 4, 4],
          'data': 'binary'}
    if xyz_rgb.dtype != np.float32:
        raise ValueError('array must be float32')
    if metadata is not None:
        md.update(metadata)
    pc_data = xyz_rgb.view(np.dtype([('x', np.float32),
                                   ('y', np.float32),
                                   ('z', np.float32),
                                   ('rgb', np.float32)])).squeeze()
    return PointCloud(md, pc_data)


def encode_rgb_for_pcl(rgb):
    """ Encode bit-packed RGB for use with PCL.
    
    Converts RGB values from separate channels to the packed float format used by PCL.
    
    Args:
        rgb: Nx3 uint8 array with RGB values (each channel 0-255).
        
    Returns:
        Nx1 float32 array with bit-packed RGB values, for PCL.
        
    Raises:
        AssertionError: If the input array is not uint8, 2D, or doesn't have 3 channels
    """
    assert(rgb.dtype == np.uint8)
    assert(rgb.ndim == 2)
    assert(rgb.shape[1] == 3)
    rgb = rgb.astype(np.uint32)
    rgb = np.array((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | (rgb[:, 2] << 0),
                   dtype=np.uint32)
    rgb.dtype = np.float32
    return rgb


def decode_rgb_from_pcl(rgb):
    """ Decode the bit-packed RGBs used by PCL.
    
    Converts PCL's packed float RGB format to separate RGB channels.
    
    Args:
        rgb: An Nx1 array of packed float RGB values.
        
    Returns:
        Nx3 uint8 array with one column per color channel (R, G, B).
    """
    rgb = rgb.copy()
    rgb.dtype = np.uint32
    r = np.asarray((rgb >> 16) & 255, dtype=np.uint8)
    g = np.asarray((rgb >> 8) & 255, dtype=np.uint8)
    b = np.asarray(rgb & 255, dtype=np.uint8)
    rgb_arr = np.zeros((len(rgb), 3), dtype=np.uint8)
    rgb_arr[:, 0] = r
    rgb_arr[:, 1] = g
    rgb_arr[:, 2] = b
    return rgb_arr


def make_xyz_label_point_cloud(xyzl, label_type='f'):
    """ Make XYZL point cloud from numpy array.
    
    Creates a point cloud from an array of XYZ coordinates and labels.
    
    Args:
        xyzl: A numpy array of shape (N, 4) containing XYZ coordinates and labels
        label_type: Type of the label field, either 'f' for float or 'u' for unsigned int
        
    Returns:
        A PointCloud object
        
    Raises:
        ValueError: If label_type is not 'f' or 'u'
    """
    md = {'version': .7,
          'fields': ['x', 'y', 'z', 'label'],
          'count': [1, 1, 1, 1],
          'width': len(xyzl),
          'height': 1,
          'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          'points': len(xyzl),
          'data': 'ASCII'}
    if label_type.lower() == 'f':
        md['size'] = [4, 4, 4, 4]
        md['type'] = ['F', 'F', 'F', 'F']
    elif label_type.lower() == 'u':
        md['size'] = [4, 4, 4, 1]
        md['type'] = ['F', 'F', 'F', 'U']
    else:
        raise ValueError('label type must be F or U')
    # TODO use .view()
    xyzl = xyzl.astype(np.float32)
    dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),
                   ('label', np.float32)])
    pc_data = np.rec.fromarrays([xyzl[:, 0], xyzl[:, 1], xyzl[:, 2],
                                 xyzl[:, 3]], dtype=dt)
    pc = PointCloud(md, pc_data)
    return pc


def point_cloud_to_ply(pc, fileobj, data_format='ascii'):
    """Write pointcloud as .ply to a file object.
    
    Args:
        pc: A PointCloud object
        fileobj: A file object opened in appropriate mode ('w' for ASCII, 'wb' for binary)
        data_format: Format to save in ('ascii' or 'binary')
        
    Note:
        Binary format uses little-endian byte order.
    """
    # Validate format
    if data_format.lower() not in ('ascii', 'binary'):
        raise ValueError("Format must be 'ascii' or 'binary'")
    
    # Write header
    header = ['ply',
             'format %s 1.0' % ('ascii' if data_format.lower() == 'ascii' else 'binary_little_endian'),
             'comment Generated by pypcd',
             'element vertex %d' % pc.points]
    
    # Map PCD types to PLY types
    type_map = {
        ('F', 4): 'float',
        ('F', 8): 'double',
        ('U', 1): 'uchar',
        ('U', 2): 'ushort',
        ('U', 4): 'uint',
        ('I', 1): 'char',
        ('I', 2): 'short',
        ('I', 4): 'int'
    }
    
    # Write property definitions
    for field, type_, size, count in zip(pc.fields, pc.type, pc.size, pc.count):
        if count == 1:
            ply_type = type_map.get((type_, size), 'float')
            header.append('property %s %s' % (ply_type, field))
        else:
            # Handle multi-count properties (like normals)
            for i in range(count):
                ply_type = type_map.get((type_, size), 'float')
                header.append('property %s %s_%d' % (ply_type, field, i))
    
    header.append('end_header')
    header = '\n'.join(header) + '\n'
    
    if isinstance(fileobj, (sio, bytes)) or (hasattr(fileobj, 'mode') and 'b' in fileobj.mode):
        fileobj.write(header.encode('ascii'))
    else:
        fileobj.write(header)
    
    if data_format.lower() == 'ascii':
        # Write ASCII data
        fmtstr = []
        for t, cnt in zip(pc.type, pc.count):
            if t == 'F':
                fmtstr.extend(['%.10f']*cnt)
            elif t in ('I', 'U'):
                fmtstr.extend(['%d']*cnt)
        fmtstr = ' '.join(fmtstr)
        
        for i in range(pc.points):
            line = []
            for field, count in zip(pc.fields, pc.count):
                if count == 1:
                    line.append(pc.pc_data[field][i].item())  # Extract scalar value
                else:
                    line.extend(x.item() for x in pc.pc_data[field][i])  # Extract scalar values
            fileobj.write(fmtstr % tuple(line) + '\n')
    else:
        # Write binary data
        # Ensure the data is in the correct byte order (little-endian)
        for field in pc.pc_data.dtype.names:
            data = pc.pc_data[field].astype(pc.pc_data.dtype[field], copy=True)
            if data.dtype.byteorder == '>':  # big-endian
                data = data.byteswap()
            fileobj.write(data.tobytes())

def save_point_cloud_ply(pc, fname, data_format='ascii'):
    """Save pointcloud to a PLY file.
    
    Args:
        pc: A PointCloud object
        fname: Path to the output PLY file
        data_format: Format to save in ('ascii' or 'binary')
    """
    mode = 'wb' if data_format.lower() == 'binary' else 'w'
    with open(fname, mode) as f:
        point_cloud_to_ply(pc, f, data_format)


class PointCloud(object):
    """ Wrapper for point cloud data.

    The variable members of this class parallel the ones used by
    the PCD metadata (and similar to PCL and ROS PointCloud2 messages).

    ``pc_data`` holds the actual data as a structured numpy array.

    The other relevant metadata variables are:

    - ``version``: Version, usually .7
    - ``fields``: Field names, e.g. ``['x', 'y' 'z']``
    - ``size``: Field sizes in bytes, e.g. ``[4, 4, 4]``
    - ``count``: Counts per field e.g. ``[1, 1, 1]``. NB: Multi-count field
      support is sketchy
    - ``width``: Number of points, for unstructured point clouds (assumed by
      most operations)
    - ``height``: 1 for unstructured point clouds (again, what we assume most
      of the time)
    - ``viewpoint``: A pose for the viewpoint of the cloud, as
      x y z qw qx qy qz, e.g. ``[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]``
    - ``points``: Number of points
    - ``type``: Data type of each field, e.g. ``['F', 'F', 'F']``
    - ``data``: Data storage format. One of ``ascii``, ``binary`` or ``binary_compressed``

    See `PCL docs <http://pointclouds.org/documentation/tutorials/pcd_file_format.php>`__
    for more information.
    """

    def __init__(self, metadata, pc_data):
        """Initialize a PointCloud object.
        
        Args:
            metadata: Dictionary containing PCD metadata
            pc_data: Numpy structured array containing the point cloud data
        """
        self.metadata_keys = metadata.keys()
        self.__dict__.update(metadata)
        self.pc_data = pc_data
        self.check_sanity()

    def get_metadata(self):
        """Get a copy of the point cloud metadata.
        
        Returns:
            Dictionary containing the point cloud metadata
        """
        metadata = {}
        for k in self.metadata_keys:
            metadata[k] = copy.copy(getattr(self, k))
        return metadata

    def check_sanity(self):
        """Check if the point cloud metadata is consistent.
        
        Verifies that:
        - Metadata is consistent
        - Number of points matches the metadata
        - Width * height equals the number of points
        - Number of fields matches the number of count and type entries
        
        Raises:
            AssertionError: If any of the checks fail
        """
        # pdb.set_trace()
        md = self.get_metadata()
        assert(_metadata_is_consistent(md))
        assert(len(self.pc_data) == self.points)
        assert(self.width*self.height == self.points)
        assert(len(self.fields) == len(self.count))
        assert(len(self.fields) == len(self.type))

    def save(self, fname):
        """Save the point cloud to a PCD file in ASCII format.
        
        A convenience method that calls save_pcd with ASCII format.
        
        Args:
            fname: Path to the output PCD file
        """
        self.save_pcd(fname, compression='ascii')

    def save_pcd(self, fname, compression=None):
        """Save the point cloud to a PCD file.

        Args:
            fname: Path to the output PCD file
        compression: Optional compression type ('ascii', 'binary', or 'binary_compressed').
            If None, uses the compression type specified in the PointCloud object

        Note:
            The 'data_compression' keyword argument is deprecated in favor of 'compression'
        """
        if compression is None:
            compression = self.data
        point_cloud_to_fileobj(self, open(fname, 'wb'), compression)

    def save_pcd_to_fileobj(self, fileobj, compression=None):
        """Save the point cloud to a file object in PCD format.

        Args:
            fileobj: A file object opened in appropriate mode ('w' for ASCII, 'wb' for binary)
            compression: Optional compression type ('ascii', 'binary', or 'binary_compressed'). 
            If None, uses the compression type specified in the PointCloud object

        Note:
            The 'data_compression' keyword argument is deprecated in favor of 'compression'
        """
        point_cloud_to_fileobj(self, fileobj, compression)

    def save_pcd_to_buffer(self, compression=None):
        """Save the point cloud to a buffer in PCD format.
        
        Args:
            compression: Optional compression type ('ascii', 'binary', or 'binary_compressed').
                         If None, uses the compression type specified in the PointCloud object.
        
        Returns:
            A buffer containing the PCD data
        
        Note:
            The 'data_compression' keyword argument is deprecated in favor of 'compression'.
        """
        fileobj = sio()
        point_cloud_to_fileobj(self, fileobj, compression)
        return fileobj.getvalue()

    def save_txt(self, fname):
        """Save the point cloud to a text file.
        
        A convenience method that calls the save_txt function.
        
        Args:
            fname: Path to the output text file
        """
        save_txt(self, fname)

    def save_xyz_label(self, fname, **kwargs):
        """Save the point cloud to a text file with XYZ coordinates and labels.

        A convenience method that calls the save_xyz_label function.

        Args:
            fname: Path to the output text file
            kwargs: Additional keyword arguments to pass to save_xyz_label
        """
        save_xyz_label(self, fname, **kwargs)

    def save_xyz_intensity_label(self, fname, **kwargs):
        """Save the point cloud to a text file with XYZ coordinates, intensity, and labels.

        A convenience method that calls the save_xyz_intensity_label function.

        Args:
            fname: Path to the output text file
            kwargs: Additional keyword arguments to pass to save_xyz_intensity_label
        """
        save_xyz_intensity_label(self, fname, **kwargs)

    def copy(self):
        """Create a deep copy of the point cloud.
        
        Returns:
            A new PointCloud object with the same metadata and data
        """
        new_pc_data = np.copy(self.pc_data)
        new_metadata = self.get_metadata()
        return PointCloud(new_metadata, new_pc_data)

    def to_msg(self):
        """Convert the point cloud to a ROS PointCloud2 message.
        
        Returns:
            A ROS sensor_msgs/PointCloud2 message
            
        Raises:
            Exception: If ROS sensor_msgs is not available
        """
        if not HAS_SENSOR_MSGS:
            raise Exception('ROS sensor_msgs not found')
        # TODO is there some metadata we want to attach?
        return numpy_pc2.array_to_pointcloud2(self.pc_data)

    @staticmethod
    def from_path(fname):
        """Create a PointCloud object from a PCD file.
        
        Args:
            fname: Path to the PCD file
            
        Returns:
            A PointCloud object
        """
        return point_cloud_from_path(fname)

    @staticmethod
    def from_fileobj(fileobj):
        """Create a PointCloud object from a file object.
        
        Args:
            fileobj: A file object opened in binary mode ('rb')
            
        Returns:
            A PointCloud object
        """
        return point_cloud_from_fileobj(fileobj)

    @staticmethod
    def from_buffer(buf):
        """Create a PointCloud object from a buffer.
        
        Args:
            buf: Buffer containing PCD data
            
        Returns:
            A PointCloud object
        """
        return point_cloud_from_buffer(buf)

    @staticmethod
    def from_array(arr):
        """Create a PointCloud object from a numpy structured array.
        
        Args:
            arr: A numpy structured array containing point cloud data
            
        Returns:
            A PointCloud object
            
        Note:
            The field names in the structured array become the field names in the point cloud.
        """
        pc_data = arr.copy()
        md = {'version': .7,
              'fields': [],
              'size': [],
              'count': [],
              'width': 0,
              'height': 1,
              'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              'points': 0,
              'type': [],
              'data': 'binary_compressed'}
        md['fields'] = list(pc_data.dtype.names)
        for field in md['fields']:
            field_dtype = pc_data.dtype.fields[field][0]
            if field_dtype not in numpy_type_to_pcd_type:
                raise ValueError(f"Unsupported dtype {field_dtype} for field {field}")
            type_, size_ = numpy_type_to_pcd_type[field_dtype]
            md['type'].append(type_)
            md['size'].append(size_)
            # TODO handle multicount
            md['count'].append(1)
        md['width'] = len(pc_data)
        md['points'] = len(pc_data)
        pc = PointCloud(md, pc_data)
        return pc

    @staticmethod
    def from_msg(msg, squeeze=True):
        """Create a PointCloud object from a ROS PointCloud2 message.
        
        Args:
            msg: A ROS sensor_msgs/PointCloud2 message
            squeeze: If True, fix when clouds get 1 as first dimension
            
        Returns:
            A PointCloud object
            
        Raises:
            NotImplementedError: If ROS sensor_msgs is not available
            
        Note:
            Fields with count > 1 are not well tested.
        """
        if not HAS_SENSOR_MSGS:
            raise NotImplementedError('ROS sensor_msgs not found')
        md = {'version': .7,
              'fields': [],
              'size': [],
              'count': [],
              'width': msg.width,
              'height': msg.height,
              'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              'points': 0,
              'type': [],
              'data': 'binary_compressed'}
        for field in msg.fields:
            md['fields'].append(field.name)
            t, s = pc2_type_to_pcd_type[field.datatype]
            md['type'].append(t)
            md['size'].append(s)
            # TODO handle multicount correctly
            if field.count > 1:
                warnings.warn('fields with count > 1 are not well tested')
            md['count'].append(field.count)
        pc_array = numpy_pc2.pointcloud2_to_array(msg)
        pc_data = pc_array.reshape(-1)
        md['height'], md['width'] = pc_array.shape
        md['points'] = len(pc_data)
        pc = PointCloud(md, pc_data)
        return pc

    def save_ply(self, fname, data_format='ascii'):
        """Save the point cloud to a PLY file.
        
        Args:
            fname: Path to the output PLY file
            data_format: Format to save in ('ascii' or 'binary')
        """
        save_point_cloud_ply(self, fname, data_format)
