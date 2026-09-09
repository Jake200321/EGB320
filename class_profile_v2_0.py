"""
Shared data structure. Both calibrate.py (which creates profiles) and
vision_system.py (which loads and uses them) import this, so that pickle
knows how to reconstruct a ClassProfile object when loading profiles.pkl.
"""


class ClassProfile:
    """Everything learned about one class from its reference images."""
    def __init__(self, name):
        self.name = name
        self.hue_low = self.hue_high = None
        self.sat_low = self.sat_high = None
        self.val_low = self.val_high = None
        self.hue_wraps = False      # true for colours (like red) that cross the 0/180 hue seam
        self.descriptors = []       # ORB descriptors, one set per positive reference image
        self.ref_contours = []      # outline of the object in each positive reference image
        self.orb_threshold = None   # min "good" ORB matches to accept -- set from negatives
        self.shape_max_dist = None  # max matchShapes distance to accept -- set from negatives
