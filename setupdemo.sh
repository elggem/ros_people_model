#!/bin/bash

catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.6m.so
catkin config --install
catkin build cv_bridge tf2_py ros_people_model

source devel/setup.bash
source install/setup.bash
