import numpy as np
class Detection(object):
    def __init__(self, ltwh, confidence, feature, class_name=None, instance_mask=None, others=None):
        # def __init__(self, ltwh, feature):
        self.ltwh = np.asarray(ltwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.class_name = class_name
        self.instance_mask = instance_mask
        self.others = others

    def get_ltwh(self):
        return self.ltwh.copy()

    def to_tlbr(self):
        ret = self.ltwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):

        ret = self.ltwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
