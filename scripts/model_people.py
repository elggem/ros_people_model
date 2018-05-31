#!/usr/bin/python
import rospy
from models.people import PeopleModel

if __name__ == "__main__":

    try:
        rospy.init_node('model_people', anonymous=True)
        node = PeopleModel()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
