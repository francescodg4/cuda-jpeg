#!/bin/bash

convert -delay 10 -loop 0 -depth 8 -size 256x256 /tmp/ripple*.gray ripple.gif

animate ripple.gif
