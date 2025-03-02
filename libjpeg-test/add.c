#include <stdio.h>
#include <stdlib.h>

int main()
{
	unsigned char *buffer = malloc(10 * 1024);
	
	FILE *instream = fopen("/tmp/output-0.bin", "rb");
	FILE *instream1 = fopen("/tmp/output-1.bin", "rb");
	FILE *instream2 = fopen("/tmp/output-2.bin", "rb");

	size_t offset = 0;
	
	fread(buffer + offset, 1621, 1, instream);
	offset += 1621;
	
	fread(buffer + offset, 1208, 1, instream1);
	offset += 1208;
	
	fread(buffer + offset, 4, 1, instream2);
	offset += 4;
	
	fwrite(buffer, offset, 1, stdout);
	

	/* free(buffer); */
	/* fclose(instream); */
}
