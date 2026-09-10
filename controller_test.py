from controller import Controller, FirmwareVersionMismatch
import unittest


class TestController(unittest.TestCase):

    def test_firmware_major_mismatch(self):
        with self.assertRaises(FirmwareVersionMismatch):
            Controller._check_firmware_version_consistency((2, 5), (3, 7, 4))

    def test_firmware_minor_mismatch(self):
        with self.assertRaises(FirmwareVersionMismatch):
            Controller._check_firmware_version_consistency((2, 5), (2, 4, 3))

    def test_firmware_perfect_match(self):
        self.assertIsNone(
            Controller._check_firmware_version_consistency((2, 4), (2, 4, 3))
        )

    def test_firmware_subsequent_match(self):
        self.assertIsNone(
            Controller._check_firmware_version_consistency((2, 4), (2, 5, 3))
        )
