import cv2
import sys


class DictValue:
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return str(self.val)


if not hasattr(cv2.dnn, "DictValue"):
    cv2.dnn.DictValue = DictValue

sys.modules["cv2"] = cv2
