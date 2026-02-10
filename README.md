# Thesis

! All the following materials has to be used in Ubuntu 24.04 with Gazebo Harmonic 8.10 !

Drone on Gazebo:

cd ~/PX4-Autopilot
make px4_sitl gz_x500_vision

QControl:
./QGroundControl.AppImage

Still Target spawn:

gz service -s /world/default/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 1000 --req 'sdf_filename: "/home/france/Desktop/aruco_marker.sdf", name: "aruco_target", pose: {position: {x: 5.0, y: -4.5, z: 0.01}, orientation: {x: 0.707, y: 0, z: 0, w: 0.707}}'

Landing automation code:
cd ~/YourDirectoryOfLandingAlgorithm
python3 landing_controller.py

Modify drone's model:

gedit ~/PX4-Autopilot/Tools/simulation/gz/models/x500_vision/model.sdf
