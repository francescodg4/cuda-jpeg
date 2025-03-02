cjpeg-libjpeg: CPPFLAGS+=-I../jpeg-6b/
cjpeg-libjpeg: cjpeg.o
	$(CXX) -o $@ $^ ../jpeg-6b/libjpeg.a

cjpeg-libjpegturbo: CPPFLAGS+=-I../libjpeg-turbo-master -I../libjpeg-turbo-master/build
cjpeg-libjpegturbo: cjpeg.o
	$(CXX) -o $@ $^ ../libjpeg-turbo-master/build/libjpeg.a
