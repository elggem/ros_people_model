import abc
import bz2
import os
import urllib
from os.path import expanduser
import time

import rospy


class Recogniser(object):
    __metaclass__ = abc.ABCMeta

    DLIB_DIR = "~/.dlib"

    def __init__(self):
        self.is_initialised = False

    @abc.abstractmethod
    def initialise(self, download=True):
        return

    def get_file_path(self, file_name):
        return expanduser("{}/{}".format(Recogniser.DLIB_DIR, file_name))

    def download_model(self, url, file_name):
        if not os.path.exists(expanduser(Recogniser.DLIB_DIR)):
            os.makedirs(expanduser(Recogniser.DLIB_DIR))

        url_file_name = url.split("/")[-1]
        url_file_path = self.get_file_path(url_file_name)
        final_file_path = self.get_file_path(file_name)

        if not os.path.isfile(url_file_path):
            rospy.loginfo("downloading: {}".format(url))
            url_opener = urllib.URLopener()
            url_opener.retrieve(url, url_file_path)
            rospy.loginfo("finished downloading: {}".format(url))

        if url_file_path.endswith(".bz2") and not os.path.isfile(final_file_path):
            rospy.loginfo("extracting: {}".format(url_file_path))
            data = bz2.BZ2File(url_file_path).read()  # get the decompressed data
            open(final_file_path, 'wb').write(data)  # write a uncompressed file
            rospy.loginfo("finished extracting: {}".format(url_file_path))

    def wait_for_model(self, file_path):
        while not rospy.is_shutdown():
            if os.path.isfile(self.get_file_path(file_path)):
                return True
            time.sleep(1)
        return False

