#! /usr/bin/env python3
#
# Show encoders value as fast as possible, in absolute and relative mode

from controller import Controller

c = Controller()
rel = c.new_relative()

while True:
    print(c.get_raw_encoder_ticks(), c.get_relative_encoder_ticks(rel))
