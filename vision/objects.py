"""
====================================================================
Objects Subsystem - Data Structures
====================================================================
"""

class DetectedObject:
    """
    Standard object format returned by all detector modules.
    """
    def __init__(self, obj_type="unknown", bounding_box=(0, 0, 0, 0), confidence=1.0):
        self.type = obj_type             # e.g., 'victim', 'obstacle', 'ramp', 'door', 'base_zone'
        self.bounding_box = bounding_box # (x, y, width, height)
        self.confidence = confidence     # Float confidence score (0.0 to 1.5)

    def __repr__(self):
        return f"DetectedObject(type='{self.type}', bbox={self.bounding_box}, conf={self.confidence:.2f})"