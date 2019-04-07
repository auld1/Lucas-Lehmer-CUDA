lucas_lehmer : main.o add.o carry.o compare.o fft_multiply.o \
				llntt.o memory.o mers.o mod.o multiply.o rand.o subtract.o untt.o
	nvcc -lgmp -lcufft -o lucas_lehmer main.o add.o carry.o compare.o \
		fft_multiply.o llntt.o memory.o mers.o mod.o multiply.o rand.o subtract.o untt.o

main.o : main.cu
	nvcc -c main.cu

add.o : add.cu add.h
	nvcc -c add.cu

carry.o : carry.cu carry.h
	nvcc -c carry.cu

compare.o : compare.cu compare.h
	nvcc -c compare.cu

fft_multiply.o : fft_multiply.cu fft_multiply.h
	nvcc -c fft_multiply.cu

llntt.o : llntt.cu llntt.h
	nvcc -c llntt.cu

memory.o : memory.cu memory.h
	nvcc -c memory.cu

mers.o : mers.cu mers.h
	nvcc -c mers.cu

mod.o : mod.cu mod.h
	nvcc -c mod.cu

multiply.o : multiply.cu multiply.h
	nvcc -c multiply.cu

rand.o : rand.cu rand.h
	nvcc -c rand.cu

subtract.o : subtract.cu subtract.h
	nvcc -c subtract.cu

untt.o : untt.cu untt.h
	nvcc -c untt.cu


clean :
	rm lucas_lehmer main.o add.o carry.o compare.o \
		fft_multiply.o llntt.o memory.o mers.o mod.o multiply.o rand.o subtract.o untt.o
