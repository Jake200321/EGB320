"""Rescue collection subsystem — Roger. Drop your mechanism code in this folder.

Nav only ever calls `egb320.interfaces.rescue.RescueInterface`: collect() / release() /
clear_rubble() / status() / abort(). The status() reply (DONE / FAILED + has_victim)
is what lets nav honestly switch the LED to red.
"""
