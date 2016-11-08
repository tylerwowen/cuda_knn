################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/cuda_knn.cu \
../src/thrust_utils.cu \
../src/utils.cu 

CPP_SRCS += \
../src/argparser.cpp \
../src/datareader.cpp \
../src/main.cpp 

OBJS += \
./src/argparser.o \
./src/cuda_knn.o \
./src/datareader.o \
./src/main.o \
./src/thrust_utils.o \
./src/utils.o 

CU_DEPS += \
./src/cuda_knn.d \
./src/thrust_utils.d \
./src/utils.d 

CPP_DEPS += \
./src/argparser.d \
./src/datareader.d \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-8.0/bin/nvcc -O3 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-8.0/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-8.0/bin/nvcc -O3 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-8.0/bin/nvcc -O3 --compile --relocatable-device-code=true -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


