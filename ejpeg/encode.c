/* /\* */
/*  * jpeg_natural_order[i] is the natural-order position of the i'th element */
/*  * of zigzag order. */
/*  * */
/*  * When reading corrupted data, the Huffman decoders could attempt */
/*  * to reference an entry beyond the end of this array (if the decoded */
/*  * zero run length reaches past the end of the block).  To prevent */
/*  * wild stores without adding an inner-loop test, we put some extra */
/*  * "63"s after the real entries.  This will cause the extra coefficient */
/*  * to be stored in location 63 of the block, not somewhere random. */
/*  * The worst case would be a run-length of 15, which means we need 16 */
/*  * fake entries. */
/*  *\/ */

/* const int jpeg_natural_order[DCTSIZE2+16] = { */
/*   0,  1,  8, 16,  9,  2,  3, 10, */
/*  17, 24, 32, 25, 18, 11,  4,  5, */
/*  12, 19, 26, 33, 40, 48, 41, 34, */
/*  27, 20, 13,  6,  7, 14, 21, 28, */
/*  35, 42, 49, 56, 57, 50, 43, 36, */
/*  29, 22, 15, 23, 30, 37, 44, 51, */
/*  58, 59, 52, 45, 38, 31, 39, 46, */
/*  53, 60, 61, 54, 47, 55, 62, 63, */
/*  63, 63, 63, 63, 63, 63, 63, 63, /\* extra entries for safety in decoder *\/ */
/*  63, 63, 63, 63, 63, 63, 63, 63 */
/* }; */


// {0, 2, 3, 4, 5, 6, 14, 30, 62, 126, 254, 510, 0 <repeats 244 times>} // code
// {2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 0 <repeats 244 times>} // size

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

typedef struct {
	unsigned char *buf;
	unsigned char *next_output_byte;
	size_t free_in_buffer;
	size_t size;
	struct {
		uint32_t put_buffer;		/* current bit-accumulation buffer */
		int put_bits;			/* # of bits now in it */
	} cur;
} stream;


int emit_byte(stream *os, unsigned char val)
{
	*(os->next_output_byte++) = val;
	os->free_in_buffer--;
	return 0;
}


/* Outputting bits to the file */

/* Only the right 24 bits of put_buffer are used; the valid bits are
 * left-justified in this part.  At most 16 bits can be passed to emit_bits
 * in one call, and we never retain more than 7 bits in put_buffer
 * between calls, so 24 bits are sufficient.
 */
bool emit_bits (stream *os, unsigned int code, int size)
{
	/* This routine is heavily used, so it's worth coding tightly. */
	register uint32_t put_buffer = code;
	register int put_bits = os->cur.put_bits;

	if (size == 0)
		return false;

	put_buffer &= (((uint32_t) 1) << size) - 1; /* mask off any extra bits in code */
  
	put_bits += size;		/* new number of bits in buffer */
  
	put_buffer <<= 24 - put_bits; /* align incoming bits */

	put_buffer |= os->cur.put_buffer; /* and merge with old buffer contents */
  
	while (put_bits >= 8) {
		unsigned char c = (unsigned char) ((put_buffer >> 16) & 0xFF);
    
		emit_byte(os, c);
		
		if (c == 0xFF)
			emit_byte(os, 0);

		put_buffer <<= 8;
		put_bits -= 8;
	}

	os->cur.put_buffer = put_buffer; /* update os variables */
	os->cur.put_bits = put_bits;

	return true;
}

#define JCOEFPTR int *
#define MAX_COEF_BITS 10 // ?
#define DCTSIZE2 64

typedef const struct {
	int ehufco[256];
	int ehufsi[256];
} c_derived_tbl;


const c_derived_tbl dctbl = {
	.ehufco = {0, 2, 3, 4, 5, 6, 14, 30, 62, 126, 254, 510, 0},
	.ehufsi = {2, 3, 3, 3, 3, 3,  4,  5,  6,   7,   8,   9, 0}
};

const c_derived_tbl actbl = {
	.ehufco = {0, 2, 3, 4, 5, 6, 14, 30, 62, 126, 254, 510, 0},
	.ehufsi = {2, 3, 3, 3, 3, 3,  4,  5,  6,   7,   8,   9, 0}
};


static const UINT8 bits_dc_luminance[17] =
{ /* 0-base */ 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
static const UINT8 val_dc_luminance[] =
{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
  
static const UINT8 bits_dc_chrominance[17] =
{ /* 0-base */ 0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
static const UINT8 val_dc_chrominance[] =
{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
  
static const UINT8 bits_ac_luminance[17] =
{ /* 0-base */ 0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d };
static const UINT8 val_ac_luminance[] =
{ 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
  0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
  0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
  0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
  0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
  0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
  0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
  0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
  0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
  0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
  0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
  0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
  0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
  0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
  0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
  0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
  0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
  0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
  0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
  0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
  0xf9, 0xfa };
  
static const UINT8 bits_ac_chrominance[17] =
{ /* 0-base */ 0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 };
static const UINT8 val_ac_chrominance[] =
{ 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
  0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
  0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
  0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
  0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
  0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
  0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
  0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
  0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
  0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
  0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
  0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
  0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
  0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
  0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
  0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
  0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
  0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
  0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
  0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
  0xf9, 0xfa };


const int jpeg_natural_order[] = {
	0,  1,  8, 16,  9,  2,  3, 10,
	17, 24, 32, 25, 18, 11,  4,  5,
	12, 19, 26, 33, 40, 48, 41, 34,
	27, 20, 13,  6,  7, 14, 21, 28,
	35, 42, 49, 56, 57, 50, 43, 36,
	29, 22, 15, 23, 30, 37, 44, 51,
	58, 59, 52, 45, 38, 31, 39, 46,
	53, 60, 61, 54, 47, 55, 62, 63,
	63, 63, 63, 63, 63, 63, 63, 63, /* extra entries for safety in decoder */
	63, 63, 63, 63, 63, 63, 63, 63
};

bool encode_one_block (
        stream * os,
	JCOEFPTR block,
	int last_dc_val,
	c_derived_tbl *dctbl,
	c_derived_tbl *actbl)
{
	register int temp, temp2;
	register int nbits;
	register int k, r, i;
  
	/* Encode the DC coefficient difference per section F.1.2.1 */
  
	temp = temp2 = block[0] - last_dc_val;

	if (temp < 0) {
		temp = -temp;		/* temp is abs value of input */
		/* For a negative input, want temp2 = bitwise complement of abs(input) */
		/* This code assumes we are on a two's complement machine */
		temp2--;
	}
  
	/* Find the number of bits needed for the magnitude of the coefficient */
	nbits = 0;
	while (temp) {
		nbits++;
		temp >>= 1;
	}
	/* Check for out-of-range coefficient values.
	 * Since we're encoding a difference, the range limit is twice as much.
	 */
	/* if (nbits > MAX_COEF_BITS+1) */
	/* 	ERREXIT(state->cinfo, JERR_BAD_DCT_COEF); */
	if (nbits > MAX_COEF_BITS+1)
		return false;
  
	/* Emit the Huffman-coded symbol for the number of bits */
	if (! emit_bits(os, dctbl->ehufco[nbits], dctbl->ehufsi[nbits]))
		return false;

	/* Emit that number of bits of the value, if positive, */
	/* or the complement of its magnitude, if negative. */
	if (nbits) {			/* emit_bits rejects calls with size 0 */
		if (! emit_bits(os, (unsigned int) temp2, nbits))
			return false;
	}

	/* Encode the AC coefficients per section F.1.2.2 */
  
	r = 0;			/* r = run length of zeros */
  
	for (k = 1; k < DCTSIZE2; k++) {
		if ((temp = block[jpeg_natural_order[k]]) == 0) {
			r++;
		} else {
			/* if run length > 15, must emit special run-length-16 codes (0xF0) */
			while (r > 15) {
				if (! emit_bits(os, actbl->ehufco[0xF0], actbl->ehufsi[0xF0]))
					return false;
				r -= 16;
			}

			temp2 = temp;
			if (temp < 0) {
				temp = -temp;		/* temp is abs value of input */
				/* This code assumes we are on a two's complement machine */
				temp2--;
			}
      
			/* Find the number of bits needed for the magnitude of the coefficient */
			nbits = 1;		/* there must be at least one 1 bit */
			while ((temp >>= 1))
				nbits++;
			/* /\* Check for out-of-range coefficient values *\/ */
			/* if (nbits > MAX_COEF_BITS) */
			/* 	ERREXIT(state->cinfo, JERR_BAD_DCT_COEF); */
      
			/* Emit Huffman symbol for run length / number of bits */
			i = (r << 4) + nbits;
			if (! emit_bits(os, actbl->ehufco[i], actbl->ehufsi[i]))
				return false;

			/* Emit that number of bits of the value, if positive, */
			/* or the complement of its magnitude, if negative. */
			if (! emit_bits(os, (unsigned int) temp2, nbits))
				return false;
      
			r = 0;
		}
	}

	/* If the last coef(s) were zero, emit an end-of-block code */
	if (r > 0)
		if (! emit_bits(os, actbl->ehufco[0], actbl->ehufsi[0]))
			return false;	

	return true;
}


int main()
{
	unsigned char buffer[1024];

	memset(buffer, 0, 1024);

	stream os = {
		.buf = buffer,
		.next_output_byte = buffer,
		.free_in_buffer = 1024,
		.size = 1024,
		.cur = {.put_buffer = 0, .put_bits = 0}
	};
	
	emit_bits(&os, 0b1101, 4);
	emit_bits(&os, 0b1001, 4);
	emit_bits(&os, 0b00, 2);
	emit_bits(&os, 0b01, 2);
	emit_bits(&os, 0xf, 4);

	int block[] = {-83, 2, 4};
	
	encode_one_block (&os, block, 0, &dctbl, &actbl);
	
	return 0;
}

#if 0
LOCAL(boolean)
encode_one_block (working_state * state, JCOEFPTR block, int last_dc_val,
		  c_derived_tbl *dctbl, c_derived_tbl *actbl)
{
	register int temp, temp2;
	register int nbits;
	register int k, r, i;
  
	/* Encode the DC coefficient difference per section F.1.2.1 */
  
	temp = temp2 = block[0] - last_dc_val;

	if (temp < 0) {
		temp = -temp;		/* temp is abs value of input */
		/* For a negative input, want temp2 = bitwise complement of abs(input) */
		/* This code assumes we are on a two's complement machine */
		temp2--;
	}
  
	/* Find the number of bits needed for the magnitude of the coefficient */
	nbits = 0;
	while (temp) {
		nbits++;
		temp >>= 1;
	}
	/* Check for out-of-range coefficient values.
	 * Since we're encoding a difference, the range limit is twice as much.
	 */
	if (nbits > MAX_COEF_BITS+1)
		ERREXIT(state->cinfo, JERR_BAD_DCT_COEF);
  
	/* Emit the Huffman-coded symbol for the number of bits */
	if (! emit_bits(state, dctbl->ehufco[nbits], dctbl->ehufsi[nbits]))
		return FALSE;

	/* Emit that number of bits of the value, if positive, */
	/* or the complement of its magnitude, if negative. */
	if (nbits)			/* emit_bits rejects calls with size 0 */
		if (! emit_bits(state, (unsigned int) temp2, nbits))
			return FALSE;

	/* Encode the AC coefficients per section F.1.2.2 */
  
	r = 0;			/* r = run length of zeros */
  
	for (k = 1; k < DCTSIZE2; k++) {
		if ((temp = block[jpeg_natural_order[k]]) == 0) {
			r++;
		} else {
			/* if run length > 15, must emit special run-length-16 codes (0xF0) */
			while (r > 15) {
				if (! emit_bits(state, actbl->ehufco[0xF0], actbl->ehufsi[0xF0]))
					return FALSE;
				r -= 16;
			}

			temp2 = temp;
			if (temp < 0) {
				temp = -temp;		/* temp is abs value of input */
				/* This code assumes we are on a two's complement machine */
				temp2--;
			}
      
			/* Find the number of bits needed for the magnitude of the coefficient */
			nbits = 1;		/* there must be at least one 1 bit */
			while ((temp >>= 1))
				nbits++;
			/* Check for out-of-range coefficient values */
			if (nbits > MAX_COEF_BITS)
				ERREXIT(state->cinfo, JERR_BAD_DCT_COEF);
      
			/* Emit Huffman symbol for run length / number of bits */
			i = (r << 4) + nbits;
			if (! emit_bits(state, actbl->ehufco[i], actbl->ehufsi[i]))
				return FALSE;

			/* Emit that number of bits of the value, if positive, */
			/* or the complement of its magnitude, if negative. */
			if (! emit_bits(state, (unsigned int) temp2, nbits))
				return FALSE;
      
			r = 0;
		}
	}

	/* If the last coef(s) were zero, emit an end-of-block code */
	if (r > 0)
		if (! emit_bits(state, actbl->ehufco[0], actbl->ehufsi[0]))
			return FALSE;

	return TRUE;
}
#endif
