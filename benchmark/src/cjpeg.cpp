#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

extern "C" {
#include "jpeglib.h"
}

#include <chrono>
#include <iostream>
#include <vector>

#define MAX_CHANNELS 3
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ROUND_UP_4(num)  (((num) + 3) & ~3)

#define CH_SUB_440 "4:4:0"
#define CH_SUB_420 "4:2:0"

typedef struct {
	int width;
	int height;
	J_COLOR_SPACE color_space;
	std::string chroma_subsampling;
} image_t;


void help(void)
{
	std::cout << "rgb2yuv <infile> <width> <height> <outfile>\n";
}


void rgb2ycbcr(uint8_t *yuv, const uint8_t *rgb, int width, int height)
{
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {			
		        float r = rgb[(i * width + j)*3 + 0];
		        float g = rgb[(i * width + j)*3 + 1];
		        float b = rgb[(i * width + j)*3 + 2];

			yuv[(i * width + j)*3 + 0] =      0.299    *r +0.587    *g +0.114    *b;
			yuv[(i * width + j)*3 + 1] = 128 -0.168736 *r -0.331264 *g +0.5      *b;
			yuv[(i * width + j)*3 + 2] = 128 +0.5      *r -0.418688 *g -0.081312 *b;
		}
	}
}



int encode(
	FILE *outstream,
	FILE *instream,
	const image_t &image,
	int quality);


int main(int argc, const char *argv[])
{
	if (argc != 5) {
		help();
		return -1;
	}
	
	const char *infile = argv[1];
	int width = atoi(argv[2]);
	int height = atoi(argv[3]);       
	const char *outfile = argv[4];

	image_t image = {
		.width = width,
		.height = height,
		.color_space = JCS_YCbCr,
		.chroma_subsampling = CH_SUB_420
	};
	
	FILE *instream = fopen(infile, "rb");
	FILE *outstream = fopen(outfile, "wb");

	encode(outstream, instream, image, 90);

	fclose(instream);
	fclose(outstream);
}


int encode(
	FILE *outstream,
	FILE *instream,
	const image_t &image,
	int quality)
{
	int i, j, k;
	int channels = 3;

	struct jpeg_compress_struct cinfo;
	
	struct jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);

	jpeg_create_compress(&cinfo);
	jpeg_suppress_tables(&cinfo, TRUE);
	
	unsigned char **line[3];

	uint32_t comp_height[MAX_CHANNELS];
	uint32_t comp_width[MAX_CHANNELS];
	uint32_t h_samp[MAX_CHANNELS];
	uint32_t v_samp[MAX_CHANNELS];
	uint32_t h_max_samp = 0;
	uint32_t v_max_samp = 0;

	unsigned char *base[MAX_CHANNELS], *end[MAX_CHANNELS];
	unsigned int stride[MAX_CHANNELS];

	J_COLOR_SPACE color_space = image.color_space;
	uint32_t width = image.width;
	uint32_t height = image.height;
	
	comp_width[0] = width;
	comp_height[0] = height;

	// Chroma subsampling 4:2:0	
	comp_width[1] = (width+1) / 2;
	comp_height[1] = (height+1) / 2;

	comp_width[2] = (width+1) / 2;
	comp_height[2] = (height+1) / 2;
	
	h_max_samp = 0;
	v_max_samp = 0;

	for (i = 0; i < channels; ++i) {
		h_samp[i] = ROUND_UP_4(comp_width[0]) / comp_width[i];
		h_max_samp = MAX(h_max_samp, h_samp[i]);
		v_samp[i] = ROUND_UP_4(comp_height[0]) / comp_height[i];
		v_max_samp = MAX(v_max_samp, v_samp[i]);
	}

	for (i = 0; i < channels; ++i) {
		h_samp[i] = h_max_samp / h_samp[i];
		v_samp[i] = v_max_samp / v_samp[i];
	}

	cinfo.image_width = width;
	cinfo.image_height = height;
	cinfo.input_components = channels;
	cinfo.in_color_space = image.color_space;

	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, quality, TRUE);
 	
	jpeg_stdio_dest(&cinfo, outstream);

	cinfo.raw_data_in = TRUE;

	jpeg_set_colorspace(&cinfo, cinfo.in_color_space);
	    
	for (i = 0; i < channels; i++) {
		cinfo.comp_info[i].h_samp_factor = h_samp[i];
		cinfo.comp_info[i].v_samp_factor = v_samp[i];
		line[i] = (unsigned char **) malloc(v_max_samp * DCTSIZE *
						    sizeof(unsigned char *));
	}

	auto reading_start = std::chrono::system_clock::now();
	
	// Read data
	size_t datasize = (width*height*channels)*sizeof(unsigned char);
	
	unsigned char *rgb_data = (unsigned char *) malloc(datasize);
    
	fread(rgb_data, datasize, 1, instream);

	auto reading_end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_reading = reading_end - reading_start;

	// RGB to YCbCr
	auto rgb2ycbcr_start = std::chrono::system_clock::now();
	
	unsigned char *data = (unsigned char *) malloc(datasize);
	rgb2ycbcr(data, rgb_data, width, height);

	auto rgb2ycbcr_end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_rgb2ycbcr = rgb2ycbcr_end - rgb2ycbcr_start;

	auto sampling_start = std::chrono::system_clock::now();
	// Split channels
	unsigned char *ch0 =
		(unsigned char *) malloc(comp_width[0]*comp_height[0]);
	unsigned char *ch1 =
		(unsigned char *) malloc(comp_width[1]*comp_height[1]);
	unsigned char *ch2 =
		(unsigned char *) malloc(comp_width[2]*comp_height[2]);
   
	unsigned char *ptr0 = ch0;
	unsigned char *ptr1 = ch1;
	unsigned char *ptr2 = ch2;

	// // Chroma subsampling YCbCr 4:2:0
	for(int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			*(ptr0++) = data[(i*width + j)*3 + 0];
		}
	}

	// Low pass filter before subsampling (simple mean)   
	for(int i = 0; i < height; i+=2) {
		for (int j = 0; j < width; j+=2) {
			int x0 = data[((i+1)*width + (j+1))*3 + 1];
			int x1 = data[((i+1)*width + j)*3 + 1];
			int x2 = data[(i*width + (j+1))*3 + 1];
			int x3 = data[(i*width + j)*3 + 1];

			*(ptr1++) = (x0+x1+x2+x3)/4;

			x0 = data[((i+1)*width + (j+1))*3 + 2];
			x1 = data[((i+1)*width + j)*3 + 2];
			x2 = data[(i*width + (j+1))*3 + 2];
			x3 = data[(i*width + j)*3 + 2];

			*(ptr2++) = (x0+x1+x2+x3)/4;
		}
	}
	
	for (i = 0; i < channels; i++) {
		if (i == 0)
			base[i] = (unsigned char *) ch0;
		if (i == 1)
			base[i] = (unsigned char *) ch1;
		if (i == 2)
			base[i] = (unsigned char *) ch2;

		uint32_t bytesperpixel = 1;
	    
		stride[i] = comp_width[i] * bytesperpixel;
		end[i] = base[i] + comp_height[i] * stride[i];
	}

	auto sampling_end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_sampling = sampling_end - sampling_start;

       	auto compression_start = std::chrono::system_clock::now();
	
	jpeg_start_compress(&cinfo, TRUE);
    
	for (i = 0; i < height; i += v_max_samp * DCTSIZE) {
		for (k = 0; k < channels; k++) {
			for (j = 0; j < v_samp[k] * DCTSIZE; j++) {
				line[k][j] = base[k];
				if (base[k] + stride[k] < end[k])
					base[k] += stride[k];
			}
		}

		jpeg_write_raw_data(&cinfo, (JSAMPIMAGE) line, v_max_samp * DCTSIZE);
	}

	jpeg_finish_compress(&cinfo);

	auto compression_end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_compression = compression_end - compression_start;	

	// Cleanup
	free(data);
	free(rgb_data);
	
	free(ch0);
	free(ch1);
	free(ch2);

	for (i = 0; i < channels; i++)
	    free(line[i]);

	double total = elapsed_reading.count() + elapsed_rgb2ycbcr.count() + elapsed_sampling.count() + elapsed_compression.count();

	std::cout << "Reading: " << elapsed_reading.count() * 1e3 << " ms "
		  << elapsed_reading.count()/total << "\n"
		  << "RGB2YCBCR: " << elapsed_rgb2ycbcr.count() * 1e3 << " ms "
		  << elapsed_rgb2ycbcr.count()/total << "\n"
		  << "Sampling: " << elapsed_sampling.count() * 1e3 << " ms "
		  << elapsed_sampling.count()/total << "\n"
		  << "Compression: " << elapsed_compression.count() * 1e3 << " ms "
		  << elapsed_compression.count()/total << "\n"
		  << "Total: " << total * 1e3 << " ms\n";

	return 0;
}
