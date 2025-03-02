#include <cstdlib>
#include <iostream>
#include <vector>
#include <numeric>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <sys/uio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#define FILENAME "/tmp/output"

template <class Container>
void init_frames(Container &frames, int nblocks, size_t capacity)
{
	std::generate_n(
		std::back_inserter(frames),
		nblocks,
		[&frames, &capacity, &nblocks]{
			unsigned char *block = (unsigned char *) malloc(capacity);
			// auto size = rand() % capacity;
			auto size = capacity;
			std::iota(block, block + size, 0);
			return ((struct iovec){ block, size });
		});
}


double os_test(int nblocks, std::size_t capacity, std::ostream &os)
{
	std::vector<struct iovec> frames;
	
	init_frames(frames, nblocks, capacity);
	
	auto begin = std::chrono::steady_clock::now();
	
	for (auto frame : frames)
		os.write((const char *) frame.iov_base, frame.iov_len);
	
	auto end = std::chrono::steady_clock::now();

	return std::chrono::duration<double, std::milli>(end - begin).count();
}


double test_vector(const int nblocks, const std::size_t capacity)
{
	std::vector<struct iovec> frames;

	init_frames(frames, nblocks, capacity);
	
	FILE *ostream = fopen(FILENAME, "wb");
	
	auto begin = std::chrono::steady_clock::now();
	
	writev(fileno(ostream), frames.data(), frames.size());		
	
	auto end = std::chrono::steady_clock::now();

	double duration = std::chrono::duration<double, std::milli>(end - begin).count();

	fclose(ostream);

	return duration;
}


double test_fstream(const int nblocks, const std::size_t capacity)
{	
	std::ofstream os(FILENAME);
	
	double duration = os_test(nblocks, capacity, os);
	
	os.close();

	return duration;	
}


double test_mmap(const int nblocks, const std::size_t capacity)
{
	const int FILESIZE = (nblocks * capacity);
	
	int fd = open(FILENAME, O_RDWR | O_CREAT, (mode_t)0600);
	
	lseek(fd, FILESIZE-1, SEEK_SET);
	if (write(fd, "", 1) < 0)
		return 0;
	
	char *buf = (char *) mmap(NULL, FILESIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
	
	std::ostringstream os;

	os.rdbuf()->pubsetbuf(buf, FILESIZE);

	double duration = os_test(nblocks, capacity, os);

	munmap(buf, FILESIZE);
	
	close(fd);

	return duration;
}


int main(int argc, const char *argv[])
{
	const int N_TRIALS = 10;
	int nblocks = 10;
	std::size_t capacity = 20;
	std::string test_type = "";

	argc--;

	if (argc == 3) {
		test_type = argv[3];
		argc--;
	}

	if (argc == 2) {
		capacity = atol(argv[2]);
		argc--;
	}
	
	if (argc == 1) {
		nblocks = atoi(argv[1]);
		argc--;
	}
	
	srand(0);

	// run tests

	std::vector<double> trials;

	double (*do_test)(const int nblocks, const std::size_t capacity);
										
	if (!test_type.compare("mmap")) {
		do_test = test_mmap;
	} else if (!test_type.compare("fstream")) {
		do_test = test_fstream;
	} else if(!test_type.compare("vector")) {
		do_test = test_vector;
	} else {
		std::cerr << "Invalid test" << std::endl;
		std::cerr << "nblocks capacity {mmap, fstream, vector}" << std::endl;
		exit(-1);
	}

	std::cerr << test_type << ", " << nblocks << ", " << capacity << std::endl;

	std::generate_n(
		std::back_inserter(trials),
		N_TRIALS,
		[&]{ return do_test(nblocks, capacity); });

	auto average = std::accumulate(std::begin(trials), std::end(trials), 0.);
	average /= trials.size();

	// estimation t = 2e-6*n*c
	
	std::cout << "average [ms]: " << average << "\n"
		  << "total, " << 2e-6*(nblocks*capacity)
		  << std::endl;
	
	return EXIT_SUCCESS;
}
