#include <npp.h>
#include <cuda_runtime.h>
#include <Exceptions.h>

#include "Endianess.h"
#include <math.h>
#include <cmath>
#include <string.h>
#include <fstream>
#include <iostream>

#include <helper_string.h>
#include <helper_cuda.h>

using namespace std;

// K.1 - suggested luminance QT
static const uint8_t default_qt_luma[] =
{
   16,11,10,16, 24, 40, 51, 61,
   12,12,14,19, 26, 58, 60, 55,
   14,13,16,24, 40, 57, 69, 56,
   14,17,22,29, 51, 87, 80, 62,
   18,22,37,56, 68,109,103, 77,
   24,35,55,64, 81,104,113, 92,
   49,64,78,87,103,121,120,101,
   72,92,95,98,112,100,103, 99,
};

/*
static const uint8_t default_qt_chroma[] =
{
    // Example QT from JPEG paper
    16,  12, 14,  14, 18, 24,  49,  72,
    11,  10, 16,  24, 40, 51,  61,  12,
    13,  17, 22,  35, 64, 92,  14,  16,
    22,  37, 55,  78, 95, 19,  24,  29,
    56,  64, 87,  98, 26, 40,  51,  68,
    81, 103, 112, 58, 57, 87,  109, 104,
    121,100, 60,  69, 80, 103, 113, 120,
    103, 55, 56,  62, 77, 92,  101, 99,
};
*/

static const uint8_t default_qt_chroma[] =
{
17,18,24,47,99,99,99,99
18,21,26,66,99,99,99,99
24,26,56,99,99,99,99,99
47,66,99,99,99,99,99,99
99,99,99,99,99,99,99,99
99,99,99,99,99,99,99,99
99,99,99,99,99,99,99,99
99,99,99,99,99,99,99,99
};



struct FrameHeader
{
    unsigned char nSamplePrecision;
    unsigned short nHeight;
    unsigned short nWidth;
    unsigned char nComponents;
    unsigned char aComponentIdentifier[3];
    unsigned char aSamplingFactors[3];
    unsigned char aQuantizationTableSelector[3];
};

struct ScanHeader
{
    unsigned char nComponents;
    unsigned char aComponentSelector[3];
    unsigned char aHuffmanTablesSelector[3];
    unsigned char nSs;
    unsigned char nSe;
    unsigned char nA;
};

struct QuantizationTable
{
    unsigned char nPrecisionAndIdentifier;
    unsigned char aTable[64];
};

struct HuffmanTable
{
    unsigned char nClassAndIdentifier;
    unsigned char aCodes[16];
    unsigned char aTable[256];
};

enum teComponentSampling
{
	YCbCr_444,
	YCbCr_440,
	YCbCr_422,
	YCbCr_420,
	YCbCr_411,
	YCbCr_410,
	YCbCr_UNKNOWN
};

int DivUp(int x, int d)
{
    return (x + d - 1) / d;
}

template<typename T>
T readAndAdvance(const unsigned char *&pData)
{
    T nElement = readBigEndian<T>(pData);
    pData += sizeof(T);
    return nElement;
}

template<typename T>
void writeAndAdvance(unsigned char *&pData, T nElement)
{
    writeBigEndian<T>(pData, nElement);
    pData += sizeof(T);
}


int nextMarker(const unsigned char *pData, int &nPos, int nLength)
{
    unsigned char c = pData[nPos++];

    do
    {
        while (c != 0xffu && nPos < nLength)
        {
            c =  pData[nPos++];
        }

        if (nPos >= nLength)
            return -1;

        c =  pData[nPos++];
    }
    while (c == 0 || c == 0x0ffu);

    return c;
}

void writeMarker(unsigned char nMarker, unsigned char *&pData)
{
    *pData++ = 0x0ff;
    *pData++ = nMarker;
}

void writeJFIFTag(unsigned char *&pData)
{
    const char JFIF_TAG[] =
    {
        0x4a, 0x46, 0x49, 0x46, 0x00,
        0x01, 0x02,
        0x00,
        0x00, 0x01, 0x00, 0x01,
        0x00, 0x00
    };

    writeMarker(0x0e0, pData);
    writeAndAdvance<unsigned short>(pData, sizeof(JFIF_TAG) + sizeof(unsigned short));
    memcpy(pData, JFIF_TAG, sizeof(JFIF_TAG));
    pData += sizeof(JFIF_TAG);
}

void loadJpeg(const char *input_file, unsigned char *&pJpegData, int &nInputLength)
{
    // Load file into CPU memory
    ifstream stream(input_file, ifstream::binary);

    if (!stream.good())
    {
        return;
    }

    stream.seekg(0, ios::end);
    nInputLength = (int)stream.tellg();
    stream.seekg(0, ios::beg);

    pJpegData = new unsigned char[nInputLength];
    stream.read(reinterpret_cast<char *>(pJpegData), nInputLength);
}

void readFrameHeader(const unsigned char *pData, FrameHeader &header)
{
    readAndAdvance<unsigned short>(pData);
    header.nSamplePrecision = readAndAdvance<unsigned char>(pData);
    header.nHeight = readAndAdvance<unsigned short>(pData);
    header.nWidth = readAndAdvance<unsigned short>(pData);
    header.nComponents = readAndAdvance<unsigned char>(pData);

    for (int c=0; c<header.nComponents; ++c)
    {
        header.aComponentIdentifier[c] = readAndAdvance<unsigned char>(pData);
        header.aSamplingFactors[c] = readAndAdvance<unsigned char>(pData);
        header.aQuantizationTableSelector[c] = readAndAdvance<unsigned char>(pData);
    }

}

void writeFrameHeader(const FrameHeader &header, unsigned char *&pData)
{
    unsigned char aTemp[128];
    unsigned char *pTemp = aTemp;

    writeAndAdvance<unsigned char>(pTemp, header.nSamplePrecision);
    writeAndAdvance<unsigned short>(pTemp, header.nHeight);
    writeAndAdvance<unsigned short>(pTemp, header.nWidth);
    writeAndAdvance<unsigned char>(pTemp, header.nComponents);

    for (int c=0; c<header.nComponents; ++c)
    {
        writeAndAdvance<unsigned char>(pTemp,header.aComponentIdentifier[c]);
        writeAndAdvance<unsigned char>(pTemp,header.aSamplingFactors[c]);
        writeAndAdvance<unsigned char>(pTemp,header.aQuantizationTableSelector[c]);
    }

    unsigned short nLength = (unsigned short)(pTemp - aTemp);

    writeMarker(0x0C0, pData);
    writeAndAdvance<unsigned short>(pData, nLength + 2);
    memcpy(pData, aTemp, nLength);
    pData += nLength;
}


void readScanHeader(const unsigned char *pData, ScanHeader &header)
{
    readAndAdvance<unsigned short>(pData);

    header.nComponents = readAndAdvance<unsigned char>(pData);

    for (int c=0; c<header.nComponents; ++c)
    {
        header.aComponentSelector[c] = readAndAdvance<unsigned char>(pData);
        header.aHuffmanTablesSelector[c] = readAndAdvance<unsigned char>(pData);
    }

    header.nSs = readAndAdvance<unsigned char>(pData);
    header.nSe = readAndAdvance<unsigned char>(pData);
    header.nA = readAndAdvance<unsigned char>(pData);
}


void writeScanHeader(const ScanHeader &header, unsigned char *&pData)
{
    unsigned char aTemp[128];
    unsigned char *pTemp = aTemp;

    writeAndAdvance<unsigned char>(pTemp, header.nComponents);

    for (int c=0; c<header.nComponents; ++c)
    {
        writeAndAdvance<unsigned char>(pTemp,header.aComponentSelector[c]);
        writeAndAdvance<unsigned char>(pTemp,header.aHuffmanTablesSelector[c]);
    }

    writeAndAdvance<unsigned char>(pTemp,  header.nSs);
    writeAndAdvance<unsigned char>(pTemp,  header.nSe);
    writeAndAdvance<unsigned char>(pTemp,  header.nA);

    unsigned short nLength = (unsigned short)(pTemp - aTemp);

    writeMarker(0x0DA, pData);
    writeAndAdvance<unsigned short>(pData, nLength + 2);
    memcpy(pData, aTemp, nLength);
    pData += nLength;
}


void readQuantizationTables(const unsigned char *pData, QuantizationTable *pTables)
{
    unsigned short nLength = readAndAdvance<unsigned short>(pData) - 2;

    while (nLength > 0)
    {
        unsigned char nPrecisionAndIdentifier = readAndAdvance<unsigned char>(pData);
        int nIdentifier = nPrecisionAndIdentifier & 0x0f;

        pTables[nIdentifier].nPrecisionAndIdentifier = nPrecisionAndIdentifier;
        memcpy(pTables[nIdentifier].aTable, pData, 64);
        pData += 64;

        nLength -= 65;
    }
}

void writeQuantizationTable(const QuantizationTable &table, unsigned char *&pData)
{
    writeMarker(0x0DB, pData);
    writeAndAdvance<unsigned short>(pData, sizeof(QuantizationTable) + 2);
    memcpy(pData, &table, sizeof(QuantizationTable));
    pData += sizeof(QuantizationTable);
}

void readHuffmanTables(const unsigned char *pData, HuffmanTable *pTables)
{
    unsigned short nLength = readAndAdvance<unsigned short>(pData) - 2;

    while (nLength > 0)
    {
        unsigned char nClassAndIdentifier = readAndAdvance<unsigned char>(pData);
        int nClass = nClassAndIdentifier >> 4; // AC or DC
        int nIdentifier = nClassAndIdentifier & 0x0f;
        int nIdx = nClass * 2 + nIdentifier;
        pTables[nIdx].nClassAndIdentifier = nClassAndIdentifier;

        // Number of Codes for Bit Lengths [1..16]
        int nCodeCount = 0;

        for (int i = 0; i < 16; ++i)
        {
            pTables[nIdx].aCodes[i] = readAndAdvance<unsigned char>(pData);
            nCodeCount += pTables[nIdx].aCodes[i];
        }

        memcpy(pTables[nIdx].aTable, pData, nCodeCount);
        pData += nCodeCount;

        nLength -= (17 + nCodeCount);
    }
}

void writeHuffmanTable(const HuffmanTable &table, unsigned char *&pData)
{
    writeMarker(0x0C4, pData);

    // Number of Codes for Bit Lengths [1..16]
    int nCodeCount = 0;

    for (int i = 0; i < 16; ++i)
    {
        nCodeCount += table.aCodes[i];
    }

    writeAndAdvance<unsigned short>(pData, 17 + nCodeCount + 2);
    memcpy(pData, &table, 17 + nCodeCount);
    pData += 17 + nCodeCount;
}


void readRestartInterval(const unsigned char *pData, int &nRestartInterval)
{
    readAndAdvance<unsigned short>(pData);
    nRestartInterval = readAndAdvance<unsigned short>(pData);
}

void printHelp()
{
    cout << "jpegNPP usage" << endl;
    cout << "   -input=srcfile.jpg     (input  file JPEG image)" << endl;
    cout << "   -output=destfile.jpg   (output file JPEG image)" << endl;
    cout << "   -width=<width>         (width of JPEG image)" << endl;
    cout << "   -height=<height>       (height of JPEG image)" << endl;
    cout << "   -quality=[1..100]      (JPEG Quality factor)" << endl;
    
}

bool printfNPPinfo(int argc, char *argv[], int cudaVerMajor, int cudaVerMinor)
{
    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    bool bVal = checkCudaCapabilities(cudaVerMajor, cudaVerMinor);
    return bVal;
}


int jpeg_encode(
	const char *outfile,
	const unsigned char *imgData,
	size_t imgLength,
	int width,
	int height,
	int quality,
	int subsampling)
{
	const char *szInputFile;
	const char *szOutputFile = outfile;
	float nScaleFactor = 1;
	NppiDCTState *pDCTState;
	
	FrameHeader oFrameHeader;
	QuantizationTable aQuantizationTables[4];
	Npp8u *pdQuantizationTables;
	cudaMalloc(&pdQuantizationTables, 64 * 4);

	HuffmanTable aHuffmanTables[4];
	HuffmanTable *pHuffmanDCTables = aHuffmanTables;
	HuffmanTable *pHuffmanACTables = &aHuffmanTables[2];
	ScanHeader oScanHeader;
	memset(&oFrameHeader,0,sizeof(FrameHeader));
	memset(aQuantizationTables,0, 4 * sizeof(QuantizationTable));
	memset(aHuffmanTables,0, 4 * sizeof(HuffmanTable));
	int nMCUBlocksH = 0;
	int nMCUBlocksV = 0;
	NppiSize aSrcActualSize[3];
	teComponentSampling eComponentSampling = YCbCr_UNKNOWN;
	
	int nRestartInterval = -1;

	NppiSize aSrcSize[3];
	Npp16s *aphDCT[3] = {0,0,0};
	Npp16s *apdDCT[3] = {0,0,0};
	Npp32s aDCTStep[3];

	Npp8u *apSrcImage[3] = {0,0,0};
	Npp32s aSrcImageStep[3];

	Npp8u *apDstImage[3] = {0,0,0};
	Npp32s aDstImageStep[3];
	NppiSize aDstSize[3];       
    
    nRestartInterval = 0;

    memset(&oFrameHeader, 0, sizeof(oFrameHeader));

    oFrameHeader.nSamplePrecision = 8;
    oFrameHeader.nWidth = width;
    oFrameHeader.nHeight = height;
    oFrameHeader.nComponents = 3;
    oFrameHeader.aComponentIdentifier[0] = 1;
    oFrameHeader.aComponentIdentifier[1] = 2;
    oFrameHeader.aComponentIdentifier[2] = 3;
    oFrameHeader.aSamplingFactors[0] = 0x22;
    oFrameHeader.aSamplingFactors[1] = 0x11;
    oFrameHeader.aSamplingFactors[2] = 0x11;
    oFrameHeader.aQuantizationTableSelector[0] = 0;
    oFrameHeader.aQuantizationTableSelector[1] = 1;
    oFrameHeader.aQuantizationTableSelector[2] = 1;

    memset(&oScanHeader, 0, sizeof(oScanHeader));
    oScanHeader.nComponents = 3;
    oScanHeader.aComponentSelector[0] = 1;
    oScanHeader.aComponentSelector[1] = 2;
    oScanHeader.aComponentSelector[2] = 3;
    oScanHeader.aHuffmanTablesSelector[0] = 0x00;
    oScanHeader.aHuffmanTablesSelector[1] = 0x11;
    oScanHeader.aHuffmanTablesSelector[2] = 0x11;		
    oScanHeader.nSs = 0;
    oScanHeader.nSe = 63;
    oScanHeader.nA = 0;

    cout << "Image Size: " << oFrameHeader.nWidth << "x" << oFrameHeader.nHeight << "x" << static_cast<int>(oFrameHeader.nComponents) << endl;
    
    // Compute channel sizes as stored in the JPEG (8x8 blocks & MCU block layout)
    for (int i = 0; i < oFrameHeader.nComponents; ++ i) {
	    nMCUBlocksV = max(nMCUBlocksV, oFrameHeader.aSamplingFactors[i] & 0x0f);
	    nMCUBlocksH = max(nMCUBlocksH, oFrameHeader.aSamplingFactors[i] >> 4 );
    }

    for (int i = 0; i < oFrameHeader.nComponents; ++ i) {
	    NppiSize oBlocks;
	    NppiSize oBlocksPerMCU = {
		    oFrameHeader.aSamplingFactors[i] >> 4,
		    oFrameHeader.aSamplingFactors[i] & 0x0f
	    };

	    aSrcActualSize[i].width = DivUp(oFrameHeader.nWidth * oBlocksPerMCU.width, nMCUBlocksH);
	    aSrcActualSize[i].height = DivUp(oFrameHeader.nHeight * oBlocksPerMCU.height, nMCUBlocksV);
				
	    oBlocks.width = (int)ceil((oFrameHeader.nWidth + 7) / 8  *
				      static_cast<float>(oBlocksPerMCU.width) / nMCUBlocksH);
	    oBlocks.width = DivUp(oBlocks.width, oBlocksPerMCU.width) * oBlocksPerMCU.width;

	    oBlocks.height = (int)ceil((oFrameHeader.nHeight + 7) / 8 *
				       static_cast<float>(oBlocksPerMCU.height) / nMCUBlocksV);
	    oBlocks.height = DivUp(oBlocks.height, oBlocksPerMCU.height) * oBlocksPerMCU.height;

	    aSrcSize[i].width = oBlocks.width * 8;
	    aSrcSize[i].height = oBlocks.height * 8;

	    // Allocate Memory
	    size_t nPitch;
	    NPP_CHECK_CUDA(cudaMallocPitch(&apdDCT[i], &nPitch, oBlocks.width * 64 * sizeof(Npp16s), oBlocks.height));
	    aDCTStep[i] = static_cast<Npp32s>(nPitch);

	    NPP_CHECK_CUDA(cudaMallocPitch(&apSrcImage[i], &nPitch, aSrcSize[i].width, aSrcSize[i].height));
	    aSrcImageStep[i] = static_cast<Npp32s>(nPitch);

	    NPP_CHECK_CUDA(cudaHostAlloc(&aphDCT[i], aDCTStep[i] * oBlocks.height, cudaHostAllocDefault));
    }
    
    // Set custom Quantization table

    aQuantizationTables[0].nPrecisionAndIdentifier = 0x00;
    aQuantizationTables[1].nPrecisionAndIdentifier = 0x01;
    aQuantizationTables[2].nPrecisionAndIdentifier = 0x10;
    aQuantizationTables[3].nPrecisionAndIdentifier = 0x11;
    
    memcpy(aQuantizationTables[0].aTable, default_qt_luma, 64);
    memcpy(aQuantizationTables[1].aTable, default_qt_chroma, 64);
    
    NPP_CHECK_NPP(nppiQuantFwdRawTableInit_JPEG_8u(
    			  aQuantizationTables[0].aTable, quality));
    NPP_CHECK_NPP(nppiQuantFwdRawTableInit_JPEG_8u(
    			  aQuantizationTables[1].aTable, quality));

    // Set default Huffman tables
    memset(aHuffmanTables, 0, sizeof(aHuffmanTables));

    aHuffmanTables[0].nClassAndIdentifier = 0x00;
    aHuffmanTables[1].nClassAndIdentifier = 0x01;
    aHuffmanTables[2].nClassAndIdentifier = 0x10;
    aHuffmanTables[3].nClassAndIdentifier = 0x11;

    // Copy DCT coefficients and Quantization Tables from host to device 
    Npp8u aZigzag[] = {
            0,  1,  5,  6, 14, 15, 27, 28,
            2,  4,  7, 13, 16, 26, 29, 42,
            3,  8, 12, 17, 25, 30, 41, 43,
            9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 51, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63
    };

    for (int i = 0; i < 4; ++i) {
        Npp8u temp[64];

        for(int k = 0 ; k < 32 ; ++ k) {
            temp[2 * k + 0] = aQuantizationTables[i].aTable[aZigzag[k +  0]];
            temp[2 * k + 1] = aQuantizationTables[i].aTable[aZigzag[k + 32]];
        }
	
        NPP_CHECK_CUDA(cudaMemcpyAsync(
			       (unsigned char *)pdQuantizationTables + i * 64,
			       temp,
			       64,
			       cudaMemcpyHostToDevice));          
    }
        

    for (int i = 0; i < 3; ++i) {
        NPP_CHECK_CUDA(cudaMemcpyAsync(
			       apdDCT[i],
			       aphDCT[i],
			       aDCTStep[i] * aSrcSize[i].height / 8,
			       cudaMemcpyHostToDevice));
    }
    
    Npp8u *pdInputImage;

    NPP_CHECK_CUDA(cudaMalloc(
			   &pdInputImage,
			   imgLength));

    NPP_CHECK_CUDA(cudaMemcpy(
			   pdInputImage,
			   imgData,
			   imgLength,
			   cudaMemcpyHostToDevice));
    
    Npp32s imageStep = aSrcSize[0].width * 3;

    // Convert to JpegYCbCr
    NPP_CHECK_NPP(nppiRGBToYCbCr420_JPEG_8u_C3P3R(
			  pdInputImage,
			  imageStep,
			  apSrcImage,
			  aSrcImageStep,
			  aSrcSize[0]));
    
    /***************************
    *
    *   Processing
    *
    ***************************/
	if(aSrcActualSize[0].width == aSrcActualSize[1].width)
	{
		if (aSrcActualSize[0].height == aSrcActualSize[1].height)
		{
			eComponentSampling = YCbCr_444;
		}
		else if (abs(static_cast<float>(aSrcActualSize[0].height - 2 * aSrcActualSize[1].height)) < 3) 
		{
			eComponentSampling = YCbCr_440;
		}
	} 
	else if (abs(static_cast<float>(aSrcActualSize[0].width - 2 * aSrcActualSize[1].width)) < 3) 
	{ 
		if (aSrcActualSize[0].height == aSrcActualSize[1].height)
		{
			eComponentSampling = YCbCr_422;
		}
		else if (abs(static_cast<float>(aSrcActualSize[0].height - 2 * aSrcActualSize[1].height)) < 3) 
		{
			eComponentSampling = YCbCr_420;
		}
	} 
	else if (abs(static_cast<float>(aSrcActualSize[0].width - 4 * aSrcActualSize[1].width)) < 3)   
	{
		if (aSrcActualSize[0].height == aSrcActualSize[1].height)
		{
			eComponentSampling = YCbCr_411;
		}
		else if (abs(static_cast<float>(aSrcActualSize[0].height - 2 * aSrcActualSize[1].height)) < 3) 
		{
			eComponentSampling = YCbCr_410;
		}
	}   
	if (eComponentSampling == YCbCr_UNKNOWN)
	{
		cout << "invalid image - Y:" << aSrcActualSize[0].width << "x" << aSrcActualSize[0].height 
			 << "  Cb:"<< aSrcActualSize[1].width << "x" << aSrcActualSize[1].height 
			 << "  Cr:"<< aSrcActualSize[2].width << "x" << aSrcActualSize[2].height;
		return EXIT_FAILURE;
	}

	// Set sampling factor for destination image.
    oFrameHeader.aSamplingFactors[1] = (1 << 4) | (oFrameHeader.aSamplingFactors[1] & 0x0f);
    oFrameHeader.aSamplingFactors[1] = (oFrameHeader.aSamplingFactors[1] & 0xf0) | 1;
    oFrameHeader.aSamplingFactors[2] = (1 << 4) | (oFrameHeader.aSamplingFactors[2] & 0x0f);
    oFrameHeader.aSamplingFactors[2] = (oFrameHeader.aSamplingFactors[2] & 0xf0) | 1;
    
    switch( eComponentSampling ) {
        case YCbCr_444 : {
            // Y
            oFrameHeader.aSamplingFactors[0] = (1 << 4) | (oFrameHeader.aSamplingFactors[0] & 0x0f);
			oFrameHeader.aSamplingFactors[0] = (oFrameHeader.aSamplingFactors[0] & 0xf0) | 1;
            break;
        }
        case YCbCr_440 : {
            // Y
            oFrameHeader.aSamplingFactors[0] = (1 << 4) | (oFrameHeader.aSamplingFactors[0] & 0x0f);
			oFrameHeader.aSamplingFactors[0] = (oFrameHeader.aSamplingFactors[0] & 0xf0) | 2;
            break;
        }
        case YCbCr_422 : {
            // Y
            oFrameHeader.aSamplingFactors[0] = (2 << 4) | (oFrameHeader.aSamplingFactors[0] & 0x0f);
			oFrameHeader.aSamplingFactors[0] = (oFrameHeader.aSamplingFactors[0] & 0xf0) | 1;
            break;
        }
        case YCbCr_420 : {
            // Y
            oFrameHeader.aSamplingFactors[0] = (2 << 4) | (oFrameHeader.aSamplingFactors[0] & 0x0f);
			oFrameHeader.aSamplingFactors[0] = (oFrameHeader.aSamplingFactors[0] & 0xf0) | 2;
            break;
        }
        case YCbCr_411 : {
            // Y
            oFrameHeader.aSamplingFactors[0] = (4 << 4) | (oFrameHeader.aSamplingFactors[0] & 0x0f);
			oFrameHeader.aSamplingFactors[0] = (oFrameHeader.aSamplingFactors[0] & 0xf0) | 1;
            break;
        }            
        case YCbCr_410 : {
            // Y
            oFrameHeader.aSamplingFactors[0] = (4 << 4) | (oFrameHeader.aSamplingFactors[0] & 0x0f);
			oFrameHeader.aSamplingFactors[0] = (oFrameHeader.aSamplingFactors[0] & 0xf0) | 2;
            break;
        }            
        default:
            return EXIT_FAILURE; 

    };

    
    // Compute channel sizes as stored in the output JPEG (8x8 blocks & MCU block layout)
    NppiSize oDstImageSize;
    float frameWidth = floor((float)oFrameHeader.nWidth * (float)nScaleFactor);
    float frameHeight = floor((float)oFrameHeader.nHeight * (float)nScaleFactor);

    oDstImageSize.width  = (int)max(1.0f, frameWidth);
    oDstImageSize.height = (int)max(1.0f, frameHeight);

    cout << "Output Size: " << oDstImageSize.width << "x" << oDstImageSize.height << "x" << static_cast<int>(oFrameHeader.nComponents) << endl;

	nMCUBlocksH = 0; 
	nMCUBlocksV = 0;
	
	for (int i=0; i < oFrameHeader.nComponents; ++i) {
		nMCUBlocksV = max(nMCUBlocksV, oFrameHeader.aSamplingFactors[i] & 0x0f );
		nMCUBlocksH = max(nMCUBlocksH, oFrameHeader.aSamplingFactors[i] >> 4 );
	}
	
	for (int i=0; i < oFrameHeader.nComponents; ++i) {
		NppiSize oBlocks;
		NppiSize oBlocksPerMCU = {oFrameHeader.aSamplingFactors[i] >> 4, oFrameHeader.aSamplingFactors[i] & 0x0f};
        
		oBlocks.width = (int)ceil((oDstImageSize.width + 7) / 8 *
					  static_cast<float>(oBlocksPerMCU.width) / nMCUBlocksH);
		oBlocks.width = DivUp(oBlocks.width, oBlocksPerMCU.width) * oBlocksPerMCU.width;

		oBlocks.height = (int)ceil((oDstImageSize.height + 7) / 8 *
					   static_cast<float>(oBlocksPerMCU.height) / nMCUBlocksV);
		oBlocks.height = DivUp(oBlocks.height, oBlocksPerMCU.height) * oBlocksPerMCU.height;

		aDstSize[i].width = oBlocks.width * 8;
		aDstSize[i].height = oBlocks.height * 8;
        
		// Allocate Memory
		size_t nPitch;
		NPP_CHECK_CUDA(cudaMallocPitch(&apDstImage[i], &nPitch, aDstSize[i].width, aDstSize[i].height));
		aDstImageStep[i] = static_cast<Npp32s>(nPitch);
	}

    // Scale to target image size
    Npp8u * apLanczosBuffer[3];
    for (int i = 0; i < 3; ++ i) {
        NppiSize oBlocksPerMCU = {oFrameHeader.aSamplingFactors[i] >> 4, oFrameHeader.aSamplingFactors[i] & 0x0f};
        NppiSize oSrcImageSize = {aSrcActualSize[i].width, aSrcActualSize[i].height};
        NppiRect oSrcImageROI = {0, 0, oSrcImageSize.width, oSrcImageSize.height};
        NppiRect oDstImageROI;
        oDstImageROI.x = 0;
        oDstImageROI.y = 0;
        oDstImageROI.width = DivUp(oDstImageSize.width * oBlocksPerMCU.width, nMCUBlocksH);
        oDstImageROI.height = DivUp(oDstImageSize.height * oBlocksPerMCU.height, nMCUBlocksV);
        NppiSize oDstImageSize = {oDstImageROI.width, oDstImageROI.height};
        
        NppiInterpolationMode eInterploationMode = NPPI_INTER_LANCZOS3_ADVANCED;   
        
        int nBufferSize;
        NPP_CHECK_NPP(nppiResizeAdvancedGetBufferHostSize_8u_C1R(oSrcImageSize, oDstImageSize, &nBufferSize, NPPI_INTER_LANCZOS3_ADVANCED));
        NPP_CHECK_CUDA(cudaMalloc(&apLanczosBuffer[i], nBufferSize));
        NPP_CHECK_NPP(nppiResizeSqrPixel_8u_C1R_Advanced(apSrcImage[i], oSrcImageSize, aSrcImageStep[i], oSrcImageROI,
                          apDstImage[i], aDstImageStep[i], oDstImageROI,
                          nScaleFactor, nScaleFactor,
                          apLanczosBuffer[i],
                          NPPI_INTER_LANCZOS3_ADVANCED));        
    }

    /***************************
    *
    *   Output
    *
    ***************************/
    NPP_CHECK_NPP(nppiDCTInitAlloc(&pDCTState));
    
    // Forward DCT
    for (int i = 0; i < 3; ++i) {
        NPP_CHECK_NPP(nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW(
			      apDstImage[i],
			      aDstImageStep[i],
			      apdDCT[i],
			      aDCTStep[i],
			      pdQuantizationTables + oFrameHeader.aQuantizationTableSelector[i] * 64,
			      aDstSize[i],
			      pDCTState));
    }

    nppiDCTFree(pDCTState);


    // Huffman Encoding
    Npp8u *pdScan;
    Npp32s nScanSize;
    nScanSize = oDstImageSize.width * oDstImageSize.height * 2;
    nScanSize = nScanSize > (4 << 20) ? nScanSize : (4 << 20);    
    NPP_CHECK_CUDA(cudaMalloc(&pdScan, nScanSize));

    Npp8u *pJpegEncoderTemp;
    size_t nTempSize;
    NPP_CHECK_NPP(nppiEncodeHuffmanGetSize(aSrcSize[0], 3, &nTempSize));
    NPP_CHECK_CUDA(cudaMalloc(&pJpegEncoderTemp, nTempSize));

    NppiEncodeHuffmanSpec *apHuffmanDCTable[3];
    NppiEncodeHuffmanSpec *apHuffmanACTable[3];

    for (int i = 0; i < 3; ++i) {
        nppiEncodeHuffmanSpecInitAlloc_JPEG(
		pHuffmanDCTables[(oScanHeader.aHuffmanTablesSelector[i] >> 4)].aCodes,
		nppiDCTable,
		&apHuffmanDCTable[i]);

	nppiEncodeHuffmanSpecInitAlloc_JPEG(
		pHuffmanACTables[(oScanHeader.aHuffmanTablesSelector[i] & 0x0f)].aCodes,
		nppiACTable,
		&apHuffmanACTable[i]);
    }
    
    Npp8u * hpCodesDC[3];
    Npp8u * hpCodesAC[3];
    Npp8u * hpTableDC[3];
    Npp8u * hpTableAC[3];
    for(int iComponent = 0; iComponent < 2; ++ iComponent) {
        hpCodesDC[iComponent] = pHuffmanDCTables[iComponent].aCodes;
        hpCodesAC[iComponent] = pHuffmanACTables[iComponent].aCodes;
        hpTableDC[iComponent] = pHuffmanDCTables[iComponent].aTable;
        hpTableAC[iComponent] = pHuffmanACTables[iComponent].aTable;
    }
    
    Npp32s nScanLength;
    NPP_CHECK_NPP(nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P3R(
			  apdDCT,
			  aDCTStep,
			  nRestartInterval,
			  oScanHeader.nSs,
			  oScanHeader.nSe,
			  oScanHeader.nA >> 4,
			  oScanHeader.nA & 0x0f,
			  pdScan,
			  &nScanLength,
			  hpCodesDC,
			  hpTableDC,
			  hpCodesAC,
			  hpTableAC,
			  apHuffmanDCTable,
			  apHuffmanACTable,
			  aDstSize,
			  pJpegEncoderTemp));

    for (int i = 0; i < 3; ++i) {
        nppiEncodeHuffmanSpecFree_JPEG(apHuffmanDCTable[i]);
        nppiEncodeHuffmanSpecFree_JPEG(apHuffmanACTable[i]);
    }

    // Write JPEG
    unsigned char *pDstJpeg = new unsigned char[nScanSize];
    unsigned char *pDstOutput = pDstJpeg;

    oFrameHeader.nWidth = oDstImageSize.width;
    oFrameHeader.nHeight = oDstImageSize.height;

    writeMarker(0x0D8, pDstOutput);
    writeJFIFTag(pDstOutput);
    writeQuantizationTable(aQuantizationTables[0], pDstOutput);
    writeQuantizationTable(aQuantizationTables[1], pDstOutput);
    writeFrameHeader(oFrameHeader, pDstOutput);
    writeHuffmanTable(pHuffmanDCTables[0], pDstOutput);
    writeHuffmanTable(pHuffmanACTables[0], pDstOutput);
    writeHuffmanTable(pHuffmanDCTables[1], pDstOutput);
    writeHuffmanTable(pHuffmanACTables[1], pDstOutput);
    writeScanHeader(oScanHeader, pDstOutput);
    NPP_CHECK_CUDA(cudaMemcpy(pDstOutput, pdScan, nScanLength, cudaMemcpyDeviceToHost));
    pDstOutput += nScanLength;
    writeMarker(0x0D9, pDstOutput);

    {
        // Write result to file.
        std::ofstream outputFile(szOutputFile, ios::out | ios::binary);
        outputFile.write(reinterpret_cast<const char *>(pDstJpeg), static_cast<int>(pDstOutput - pDstJpeg));
    }

    // Cleanup
    delete [] pDstJpeg;

    cudaFree(pdInputImage);
    cudaFree(pJpegEncoderTemp);
    cudaFree(pdQuantizationTables);
    cudaFree(pdScan);

    for (int i = 0; i < 3; ++i) {
        cudaFree(apdDCT[i]);
        cudaFreeHost(aphDCT[i]);
        cudaFree(apSrcImage[i]);
        cudaFree(apDstImage[i]);
        cudaFree(apLanczosBuffer[i]);
    }

    return EXIT_SUCCESS;
}


int main(int argc, char **argv)
{
	char *inputFile, *outputFile;
	int width, height, quality;

	if (!(checkCmdLineFlag(argc, (const char **)argv, "input") &&
	      checkCmdLineFlag(argc, (const char **)argv, "output") &&
	      checkCmdLineFlag(argc, (const char **)argv, "width") &&
	      checkCmdLineFlag(argc, (const char **)argv, "height") &&
	      checkCmdLineFlag(argc, (const char **)argv, "quality"))) {
		printHelp();
		return EXIT_FAILURE;
	}

	getCmdLineArgumentString(argc, (const char **)argv, "input", (char **)&inputFile);
	getCmdLineArgumentString(argc, (const char **)argv, "output", (char **)&outputFile);
	width = getCmdLineArgumentInt(argc, (const char **)argv, "width");
	height = getCmdLineArgumentInt(argc, (const char **)argv, "height");
	quality = getCmdLineArgumentInt(argc, (const char **)argv, "quality");
	
	unsigned char *imgData;
	int imgLength;
    	
	loadJpeg(inputFile, imgData, imgLength);

	jpeg_encode(outputFile, imgData, imgLength, width, height, quality, YCbCr_420);
	
	return EXIT_SUCCESS;
}

