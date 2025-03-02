#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "jpeglib.h"

#include <sys/mman.h>
#include <chrono>
#include <iostream>

#define MAX_CHANNELS 3
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ROUND_UP_4(num)  (((num) + 3) & ~3)

void cuda_init(void)
{
	cudaFree(0);
}

int encode(void);

int main()
{
	auto start = std::chrono::system_clock::now();
	
	cuda_init();

	auto end = std::chrono::system_clock::now();

	// std::chrono::duration<double> elapsed_seconds = end - start;    
	std::cout << "Initialization time: "
		  << (1000 * (std::chrono::duration<double>(end - start)).count())
		  << "ms \n";


	for(int i = 0; i < 10; i++) {
		auto start = std::chrono::system_clock::now();
		
		encode();

		auto end = std::chrono::system_clock::now();

		std::chrono::duration<double> elapsed_seconds = end - start;
    
		std::cout << (1000 * elapsed_seconds.count()) << "ms \n";
	}
}

int encode(void)
{
	struct jpeg_compress_struct cinfo;
	int quality = 90;
	int i, j, k;
	int channels = 3;
	
/* 	 ctx.jpegenc = NvJPEGEncoder::createJPEGEncoder("jpenenc"); */


/* 	NvJPEGEncoder::NvJPEGEncoder(const char *comp_name) */
/*     :NvElement(comp_name, valid_fields) */
/* { */
	memset(&cinfo, 0, sizeof(cinfo));

	struct jpeg_error_mgr jerr;

	memset(&jerr, 0, sizeof(jerr));
	cinfo.err = jpeg_std_error(&jerr);

	jpeg_create_compress(&cinfo);
	jpeg_suppress_tables(&cinfo, TRUE);
/* } */

	
/*     TEST_ERROR(!ctx.jpegenc, "Could not create Jpeg Encoder", cleanup); */

/*     ctx.jpegenc->setCropRect(ctx.crop_left, ctx.crop_top, */
/*             ctx.crop_width, ctx.crop_height); */

/*     if(ctx.scaled_encode) */
/*     { */
/*       ctx.jpegenc->setScaledEncodeParams(ctx.scale_width, ctx.scale_height); */
/*     } */

/*     /\** */
/*      * Case 1: */
/*      * Read YUV420 image from file system to CPU buffer, encode by */
/*      * encodeFromBuffer() then write to file system. */
/*      *\/ */
/*     if (!ctx.use_fd) */
/*     { */
/*         unsigned long out_buf_size = ctx.in_width * ctx.in_height * 3 / 2; */
/*         unsigned char *out_buf = new unsigned char[out_buf_size]; */

/*         NvBuffer buffer(V4L2_PIX_FMT_YUV420M, ctx.in_width, */
/*                 ctx.in_height, 0); */

/*         buffer.allocateMemory(); */

/*         ret = read_video_frame(ctx.in_file, buffer); */
/*         TEST_ERROR(ret < 0, "Could not read a complete frame from file", */
/*                 cleanup); */

/*         ret = ctx.jpegenc->encodeFromBuffer(buffer, JCS_YCbCr, &out_buf, */
/*                 out_buf_size, ctx.quality); */

/* int */
/* NvJPEGEncoder::encodeFromBuffer(NvBuffer & buffer, J_COLOR_SPACE color_space, */
/*         unsigned char **out_buf, unsigned long &out_buf_size, */
/*         int quality) */
/* { */
	J_COLOR_SPACE color_space = JCS_RGB;
	unsigned char **line[3];

	uint32_t comp_height[MAX_CHANNELS];
	uint32_t comp_width[MAX_CHANNELS];
	uint32_t h_samp[MAX_CHANNELS];
	uint32_t v_samp[MAX_CHANNELS];
	uint32_t h_max_samp = 0;
	uint32_t v_max_samp = 0;
/*     uint32_t channels; */

	unsigned char *base[MAX_CHANNELS], *end[MAX_CHANNELS];
	unsigned int stride[MAX_CHANNELS];

#ifdef YCBCR       
	uint32_t width = 1920;
	uint32_t height = 1080;
#else
	uint32_t width = 2592;
	uint32_t height = 1920;
#endif


/*     uint32_t i, j, k; */
/*     uint32_t buffer_id; */

/*     buffer_id = profiler.startProcessing(); */

/*     jpeg_mem_dest(&cinfo, out_buf, &out_buf_size); */
/*     width = buffer.planes[0].fmt.width; */
/*     height = buffer.planes[0].fmt.height; */

    /* switch (color_space) */
    /* { */
    /*     case JCS_YCbCr: */
            channels = 3;

            comp_width[0] = width;
            comp_height[0] = height;

#ifdef YCBCR
	    comp_width[1] = (width+1) / 2;
            comp_height[1] = (height+1) / 2;

            comp_width[2] = (width+1) / 2;
            comp_height[2] = (height+1) / 2;
#else
	    comp_width[1] = width;
            comp_height[1] = height;

            comp_width[2] = width;
            comp_height[2] = height;

#endif
	    

    /*         break; */
    /*     default: */
    /*         COMP_ERROR_MSG("Color format " << color_space << */
    /*                        " not supported\n"); */
    /*         return -1; */
    /* } */

/*     if (channels != buffer.n_planes) */
/*     { */
/*         COMP_ERROR_MSG("Buffer not in proper format"); */
/*         return -1; */
/*     } */

    for (i = 0; i < channels; i++)
    {
	/*     if (comp_width[i] != buffer.planes[i].fmt.width || */
        /*     comp_height[i] != buffer.planes[i].fmt.height) */
        /* { */
        /*     COMP_ERROR_MSG("Buffer not in proper format"); */
        /*     return -1; */
        /* } */
    }

    h_max_samp = 0;
    v_max_samp = 0;

    for (i = 0; i < channels; ++i)
    {
        h_samp[i] = ROUND_UP_4(comp_width[0]) / comp_width[i];
        h_max_samp = MAX(h_max_samp, h_samp[i]);
        v_samp[i] = ROUND_UP_4(comp_height[0]) / comp_height[i];
        v_max_samp = MAX(v_max_samp, v_samp[i]);
    }

    for (i = 0; i < channels; ++i)
    {
        h_samp[i] = h_max_samp / h_samp[i];
        v_samp[i] = v_max_samp / v_samp[i];
    }

	cinfo.image_width = width;
	cinfo.image_height = height;
	cinfo.input_components = channels;
	cinfo.in_color_space = color_space;

	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, quality, TRUE);

	FILE *outstream = fopen("out.jpeg", "wb");
	
	jpeg_stdio_dest(&cinfo, outstream);
	
/*     jpeg_set_hardware_acceleration_parameters_enc(&cinfo, TRUE, out_buf_size, 0, 0); */
    cinfo.raw_data_in = TRUE;

    if (cinfo.in_color_space == JCS_RGB)
        jpeg_set_colorspace(&cinfo, JCS_RGB);

/*     switch (color_space) */
/*     { */
/*         case JCS_YCbCr: */
#ifdef YCBCR
    cinfo.in_color_space = JCS_YCbCr;
    jpeg_set_colorspace(&cinfo, JCS_YCbCr);
#endif

/*             break; */
/*         default: */
/*             COMP_ERROR_MSG("Color format " << color_space << " not supported\n"); */
/*             return -1; */
/*     } */

    for (i = 0; i < channels; i++)
    {
        cinfo.comp_info[i].h_samp_factor = h_samp[i];
        cinfo.comp_info[i].v_samp_factor = v_samp[i];
        line[i] = (unsigned char **) malloc(v_max_samp * DCTSIZE *
                sizeof(unsigned char *));
    }

#ifdef YCBCR
    FILE *instream = fopen("ycbcr.bin", "rb");
#else
    FILE *instream = fopen("raw.bin", "rb");
#endif

    // unsigned char *data = (unsigned char *) malloc(width*height*channels);

    /* unsigned char *data = (unsigned char *) mmap(NULL, */
    /* 						 width*height*channels, */
    /* 						 PROT_READ|PROT_WRITE, */
    /* 						 MAP_PRIVATE|MAP_ANONYMOUS, */
    /* 						 fileno(instream), */
    /* 						 0); */

    unsigned char *data = (unsigned char *) mmap(NULL,
						 width*height*channels,
						 PROT_READ,
						 MAP_PRIVATE|MAP_ANONYMOUS,
						 fileno(instream),
						 0);

    
    /* if (!fread(data, width*height*channels, 1, instream)) */
    /* 	    exit(-1); */   

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

    // Chroma subsampling YUV 4:2:0
#ifdef YCBCR
    for(int i = 0; i < height; i++) {
    	    for (int j = 0; j < width; j++) {
    		    *(ptr0++) = *(data + (j + i * width)*3 + 0);
    	    }
    }
    
    for(int i = 0; i < height; i+=2) {
    	    for (int j = 0; j < width; j+=2) {
    		    *(ptr1++)= *(data + (j + i * width)*3 + 1);
    		    *(ptr2++) = *(data + (j + i * width)*3 + 2);
    	    }
    }
#else
    for(int i = 0; i < height; i++) {
    	    for (int j = 0; j < width; j++) {
    		    *(ptr0++) = *(data + (j + i * width)*3 + 0);
    		    *(ptr1++) = *(data + (j + i * width)*3 + 1);
    		    *(ptr2++) = *(data + (j + i * width)*3 + 2);
    	    }
    }
#endif   


    for (i = 0; i < channels; i++)
    {
	    /* base[i] = (unsigned char *) buffer.planes[i].data; */
	    /* stride[i] = buffer.planes[i].fmt.stride; */
	    /* end[i] = base[i] + comp_height[i] * stride[i]; */

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

    // Start encoding    
    jpeg_start_compress(&cinfo, TRUE);
    
/*     if (cinfo.err->msg_code) */
/*     { */
/*         char err_string[256]; */
/*         cinfo.err->format_message((j_common_ptr) &cinfo, err_string); */
/*         COMP_ERROR_MSG ("Error in jpeg_start_compress: " << err_string); */
/*         return -1; */
/*     } */

    for (i = 0; i < height; i += v_max_samp * DCTSIZE)
    {
        for (k = 0; k < channels; k++)
        {
            for (j = 0; j < v_samp[k] * DCTSIZE; j++)
            {
                line[k][j] = base[k];
                if (base[k] + stride[k] < end[k])
                    base[k] += stride[k];
            }
        }
        jpeg_write_raw_data(&cinfo, line, v_max_samp * DCTSIZE);
    }

    jpeg_finish_compress(&cinfo);   

    // Cleanup

    munmap(data, width*height*channels);

    free(ch0);
    free(ch1);
    free(ch2);

    for (i = 0; i < channels; i++)
        free(line[i]);

    fclose(instream);
    fclose(outstream);


    
/*     COMP_DEBUG_MSG("Succesfully encoded Buffer"); */

/*     profiler.finishProcessing(buffer_id, false); */

/*     return 0; */
/* } */

	
/*         TEST_ERROR(ret < 0, "Error while encoding from buffer", cleanup); */

/*         ctx.out_file->write((char *) out_buf, out_buf_size); */
/*         delete[] out_buf; */

/*         goto cleanup; */
/*     } */

/* cleanup: */
/*     if (ctx.conv && ctx.conv->isInError()) */
/*     { */
/*         cerr << "VideoConverter is in error" << endl; */
/*         error = 1; */
/*     } */

/*     if (ctx.got_error) */
/*     { */
/*         error = 1; */
/*     } */

/*     delete ctx.in_file; */
/*     delete ctx.out_file; */
/*     /\** */
/*      * Destructors do all the cleanup, unmapping and deallocating buffers */
/*      * and calling v4l2_close on fd */
/*      *\/ */
/*     delete ctx.conv; */
/*     delete ctx.jpegenc; */

/*     free(ctx.in_file_path); */
/*     free(ctx.out_file_path); */

/*     return -error; */


    /* 	cleanup: */
    /* if (ctx.conv && ctx.conv->isInError()) */
    /* { */
    /*     cerr << "VideoConverter is in error" << endl; */
    /*     error = 1; */
    /* } */

    /* if (ctx.got_error) */
    /* { */
    /*     error = 1; */
    /* } */

    /* delete ctx.in_file; */
    /* delete ctx.out_file; */
    /* /\** */
    /*  * Destructors do all the cleanup, unmapping and deallocating buffers */
    /*  * and calling v4l2_close on fd */
    /*  *\/ */
    /* delete ctx.conv; */
    /* delete ctx.jpegenc; */

    /* free(ctx.in_file_path); */
    /* free(ctx.out_file_path); */

    /* return -error; */

	return 0;
}
