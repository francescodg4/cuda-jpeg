#!/bin/bash

# TEGRA ENVIRONMENT

if [ $(uname -r) == "4.9.140-tegra" ]; then
    export C_INCLUDE_PATH=$(dirname $(find /usr -name "jpeglib.h"))
    ln -sf /usr/lib/aarch64-linux-gnu/libjpeg.so.8 libjpeg.so
    export LIBRARY_PATH=.
    export PATH=$PATH:/usr/local/cuda-10.0/bin
    TEGRA=1
fi


CMAKE=$(which cmake)
NVCC=$(which nvcc)

if [ -z $CMAKE ]; then
    echo "No cmake"
fi

if [ -z $NVCC ]; then
    echo "No nvcc"
fi


# Test builtin libjpeg-turbo

ldconfig -p | grep libjpeg.so

cat << EOF > test.c
#include <stdlib.h>
#include <stdio.h>
#include <jpeglib.h>

int main()
{
	printf("JPEG_LIB_VERSION %d\n", JPEG_LIB_VERSION);
	return 0;
}
EOF

echo -e "\nTest builtin libjpeg"
gcc -o test -O3 test.c -ljpeg && ./test && rm -rf test

# Test libjpeg-turbo

if [ -z $CMAKE ]; then
    echo -e "\nUnable to run libjpeg-turbo"
else
    if [ ! -f libjpeg-turbo-master/build/libjpeg.a ]; then
	unzip -n libjpeg-turbo.zip
	pushd libjpeg-turbo-master
	mkdir -p build
	cd build
	$CMAKE ../ && CFLAGS="-O3" make
	popd
    fi

    echo -e "\nTest libjpeg turbo"
    gcc -o test -O3 -Ilibjpeg-turbo-master/ test.c libjpeg-turbo-master/build/libjpeg.a && ./test rm -rf test
fi


# Test libjpeg

if [ ! -f jpeg-6b/libjpeg.a ]; then
    # Recompile library
    unzip jpeg-6b.zip
    pushd jpeg-6b
    cp ../jconfig.cfg .
    ./configure CFLAGS="-O3"
    make
    popd
fi

echo -e "\nTest libjpeg"
gcc -o test -O3 -Ijpeg-6b/ test.c jpeg-6b/libjpeg.a && ./test && rm -rf test

# Clean up
rm -rf test.c

# Test, if nvidia exists (Tegra, Jetson)
echo -e "\nTest nvidia library"
ldconfig -p | grep nvjpeg

echo $TEGRA

if [ ! -z $NVCC ]; then
    cat << EOF > main.cu
#include <stdlib.h>
#include <stdio.h>
#include <jpeglib.h>

int main()
{
	printf("JPEG_LIB_VERSION %d\n", JPEG_LIB_VERSION);
#ifdef TEGRA_ACCELERATE
	printf("TEGRA ACCELERATE %d\n", 1);
#endif
	
	return 0;
}

EOF
    if [ ! -z $TEGRA ]; then
    	INCLUDE_PATH=$(dirname $(find /usr -name "jpeglib.h"))
    	LIB_PATH="/usr/lib/aarch64-linux-gnu/tegra"
    	$NVCC -o test -O3 -I$INCLUDE_PATH -L$LIB_PATH main.cu -lnvjpeg && ./test && rm -rf test
    else
    	$NVCC -o test -O3 main.cu -lnvjpeg && ./test && rm -rf test main.cu
    fi
    
else
    echo "Unable to find nvcc"
fi
    
# Test memory between 1M-20M

# Sweep memory test

function clean {   
    rm -rf jpeg-6b libjpeg-turbo-master
}
