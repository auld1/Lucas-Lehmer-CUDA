NVCC_CFLAGS = -O3

lucas_lehmer : main.o add.o carry.o compare.o crt_ntt.o fft_multiply.o \
				llntt.o memory.o mers.o mod.o multiply.o rand.o subtract.o untt.o
	nvcc $(NVCC_CFLAGS) -lgmp -lcufft -o lucas_lehmer main.o add.o carry.o compare.o crt_ntt.o \
		fft_multiply.o llntt.o memory.o mers.o mod.o multiply.o rand.o subtract.o untt.o

main.o : main.cu
	nvcc $(NVCC_CFLAGS) -c main.cu

add.o : add.cu add.h
	nvcc $(NVCC_CFLAGS) -c add.cu

carry.o : carry.cu carry.h
	nvcc $(NVCC_CFLAGS) -c carry.cu

compare.o : compare.cu compare.h
	nvcc $(NVCC_CFLAGS) -c compare.cu

crt_ntt.o : crt_ntt.cu crt_ntt.h
	nvcc $(NVCC_CFLAGS) -c crt_ntt.cu

fft_multiply.o : fft_multiply.cu fft_multiply.h
	nvcc $(NVCC_CFLAGS) -c fft_multiply.cu

llntt.o : llntt.cu llntt.h
	nvcc $(NVCC_CFLAGS) -c llntt.cu

memory.o : memory.cu memory.h
	nvcc $(NVCC_CFLAGS) -c memory.cu

mers.o : mers.cu mers.h
	nvcc $(NVCC_CFLAGS) -c mers.cu

mod.o : mod.cu mod.h
	nvcc $(NVCC_CFLAGS) -c mod.cu

multiply.o : multiply.cu multiply.h
	nvcc $(NVCC_CFLAGS) -c multiply.cu

rand.o : rand.cu rand.h
	nvcc $(NVCC_CFLAGS) -c rand.cu

subtract.o : subtract.cu subtract.h
	nvcc $(NVCC_CFLAGS) -c subtract.cu

untt.o : untt.cu untt.h
	nvcc $(NVCC_CFLAGS) -c untt.cu


clean :
	rm lucas_lehmer main.o add.o carry.o compare.o crt_ntt.o \
		fft_multiply.o llntt.o memory.o mers.o mod.o multiply.o rand.o subtract.o untt.o
