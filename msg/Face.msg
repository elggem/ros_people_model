# Message type used for a face

# Global fields:
Header                    header

# Required fields:
sensor_msgs/Image         crop

# World coordinates:
geometry_msgs/Point       position
float32                   certainty

# Attributes
geometry_msgs/Point[]     shapes
string                    face_id
float32[]                 emotions
float32[]                 eye_states
