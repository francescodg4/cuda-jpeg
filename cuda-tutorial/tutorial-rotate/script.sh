#!/bin/bash

wget -O /tmp/lena.png "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"

convert /tmp/lena.png -depth 8 gray:lena.gray

# display -depth 8 -size 512x512 gray:lena.gray
