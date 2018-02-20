# Message type used for raw visual parts
# of the scene.

# Global fields:
Header            header

# Required fields:
sensor_msgs/Image image
sensor_msgs/Image[] crops
sensor_msgs/RegionOfInterest[] rois

# Optional fields:
geometry_msgs/Point[] shape
string                face_id
float32[]             emotions
