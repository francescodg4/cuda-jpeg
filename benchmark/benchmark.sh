#!/bin/bash

sys=acer
width=2592
# height=1920
height=1944

function bench_printsys {
    echo -e "$(lscpu)\n"
}

function bench_genraw {
    dd if=/dev/urandom of=raw.bin bs=3 count=$(("width*height"))
}

function rgb_to_bin {
   convert test.bmp RGB:test.bin
}

function bench_genimage {
    convert -depth 8 -size "$width"x"$height" RGB:raw.bin BMP3:test.bmp
    convert -depth 8 -size "$width"x"$height" -quality 100 RGB:raw.bin test.jpeg

    identify test.bmp
}

function bench_run {
    # bash benchmark.sh | sed '$s/$/\n/' >> 2020-01-22_acer.txt
    

    outfile=$(date +%F)_$sys.txt
    
    bench_printsys > $outfile    

    echo "Test image:" >> $outfile
    
    bench_genimage >> $outfile
    echo "" >> $outfile

    echo "running benchmarks..."
    for i in $(seq 1 10); do
	echo -e "\nrunning $i/10"
	run_benchmarks | tee /dev/tty | sed '$s/$/\n/' >> $outfile
    done
    
    cat $outfile
}

function run_benchmarks {

    /usr/bin/time -f %e convert -size "$width"x"$height" -depth 8 -quality 90 RGB:raw.bin JPEG:/dev/null 2>&1 \
	| awk '{print $1"*1000"}' | bc \
	| awk '{print "imagemagick, "$1" ms"}'

    /usr/bin/time -f %e ffmpeg -y -loglevel quiet -i test.bmp out.jpeg 2>&1 \
	| awk '{print $1"*1000"}' | bc \
	| awk '{print "ffmpeg, "$1" ms"}'
    rm out.jpeg

    /usr/bin/time -f %e ./cjpeg.o2 -dct fast -quality 90 -outfile /dev/null test.bmp 2>&1 \
	| awk '{print $1"*1000"}' | bc \
	| awk '{print "cjpeg.o2, "$1" ms"}'

    /usr/bin/time -f %e ./cjpeg.o3 -dct fast -quality 90 -outfile /dev/null test.bmp 2>&1 \
	| awk '{print $1"*1000"}' | bc \
	| awk '{print "cjpeg.o3, "$1" ms"}'

    /usr/bin/time -f %e ./cjpeg-libjpeg-turbo -dct fast -quality 90 -outfile /dev/null test.bmp 2>&1 \
	| awk '{print $1"*1000"}' | bc \
	| awk '{print "libjpeg-turbo, "$1" ms"}'
}


# bench_run
