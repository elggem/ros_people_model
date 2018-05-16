# Message type used for raw visual parts
# of the scene.

# Global fields:
Header            header

# Required fields:
sensor_msgs/Image crop
sensor_msgs/RegionOfInterest roi

# Optional fields
geometry_msgs/Point[]     shapes
string                    face_id
float32[]                 emotions
float32[]                 eye_states
