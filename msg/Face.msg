# Message type used for a face

# Global fields:
Header                    header

# Required fields:
sensor_msgs/Image         crop

# World coordinates:
geometry_msgs/Point      global_coordinate
float32                   certainty

# Attributes
geometry_msgs/Point[]     shapes
string                    face_ids
float32[]                 emotions
float32[]                 eyes_closed
