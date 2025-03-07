``pypcd``
=========

**This is a fork of the original ``pypcd`` library to support Python 3.**

What?
-----

Pure Python module to read and write point clouds stored in the
`PCD file format <http://pointclouds.org/documentation/tutorials/pcd_file_format.php>`__,
used by the `Point Cloud Library <http://pointclouds.org/>`__.

Why?
----

You want to mess around with your point cloud data without writing C++
and waiting hours for the template-heavy PCL code to compile.

You tried to get some of the Python bindings for PCL to compile
and just gave up.

How does it work?
-----------------

It parses the PCD header and loads the data (whether in ``ascii``,
``binary`` or ``binary_compressed`` format) as a
`Numpy <http://www.numpy.org>`__ structured array. It creates an
instance of the ``PointCloud`` class, containing the point cloud data as ``pc_data``, and
some convenience functions for I/O and metadata access.
See the comments in ``pypcd.py`` for some info on the point cloud
structure.

Example
-------

.. code:: python

    import pypcd
    # also can read from file handles.
    pc = pypcd.PointCloud.from_path('foo.pcd')
    # pc.pc_data has the data as a structured array
    # pc.fields, pc.count, etc have the metadata

    # center the x field
    pc.pc_data['x'] -= pc.pc_data['x'].mean()

    # save as binary compressed
    pc.save_pcd('bar.pcd', compression='binary_compressed')


Installation
------------

Using pip (recommended):

.. code:: bash

    pip install pypcd

Optional dependencies:
    - `pandas <https://pandas.pydata.org>`__ for DataFrame support
    - `python-lzf` for compressed PCD support

From source:

.. code:: bash

    git clone https://github.com/juhasch/pypcd
    cd pypcd
    pip install -e .

Note: Downloading data assets requires `git-lfs <https://git-lfs.github.com>`__.

ROS Integration
---------------

This library can be used with ROS ``sensor_msgs``. You can install it using pip
or integrate it into your ROS workspace:

.. code:: bash

    # Install required dependency
    pip install python-lzf
    
    # Setup in ROS workspace
    cd your_workspace/src
    git clone https://github.com/juhasch/pypcd
    mv setup_ros.py setup.py
    catkin build pypcd
    source ../devel/setup.bash

Example ROS usage:

.. code:: python

    import pypcd
    import rospy
    from sensor_msgs.msg import PointCloud2

    def cb(msg):
        pc = pypcd.PointCloud.from_msg(msg)
        pc.save('foo.pcd', compression='binary_compressed')
        # Manipulate your pointcloud
        pc.pc_data['x'] *= -1
        outmsg = pc.to_msg()
        # Set the header
        outmsg.header = msg.header
        pub.publish(outmsg)

    # ROS node setup
    sub = rospy.Subscriber('incloud', PointCloud2)
    pub = rospy.Publisher('outcloud', PointCloud2, cb)
    rospy.init_node('pypcd_node')
    rospy.spin()

Features
--------

- Supports ``ascii``, ``binary`` and ``binary_compressed`` PCD formats
- RGB encoding/decoding to single ``float32``
- Conversion to/from pandas DataFrames
- ROS PointCloud2 message conversion
- Python 3 support

Known Limitations
-----------------

- No automatic synchronization between metadata fields and ``pc_data``
- Primary focus on unorganized point clouds (``height=1``)
- Limited testing for padding and multi-count fields

TODO
----

- [ ] Better API for common operations
- [ ] Code cleanup and modernization
- [ ] CLI for file type conversion
- [ ] Improved structured point cloud support
- [ ] Expanded test coverage
- [ ] Enhanced documentation and examples
- [ ] Better handling of padding and multicount fields
- [ ] Improved RGB support
- [ ] PLY format export
- [ ] Package data assets in PyPI

Credits
-------

- Original code by Daniel Maturana (``dimatura@cmu.edu``)
- ROS integration code (``numpy_pc2.py``) by Jon Binney under BSD license
- Contributions from @wkentaro and the community
- Compressed point cloud implementation inspired by `Matlab PCL <https://www.mathworks.com/matlabcentral/fileexchange/40382-matlab-to-point-cloud-library>`__

License
-------

Copyright (C) 2015-2017 Daniel Maturana
