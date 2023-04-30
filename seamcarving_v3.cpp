#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <algorithm>

#define FILTER_WIDTH 3
__constant__ int dc_xFilter[FILTER_WIDTH * FILTER_WIDTH];
__constant__ int dc_yFilter[FILTER_WIDTH * FILTER_WIDTH];

#define CHECK(call){\
  const cudaError_t error = call;\
  if (error != cudaSuccess){\
    fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
    fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));\
    exit(EXIT_FAILURE);\
  }\
}

struct GpuTimer{

    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer(){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer(){
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start(){
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop(){
        cudaEventRecord(stop, 0);
    }

    float Eplapsed(){
        float eplapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&eplapsed, start, stop);

        return eplapsed;
    }
};

void readPnm (char *fileName, int &width, int &height, uchar3 *&pixels){
    FILE *f = fopen(fileName, "r");

    if (f == NULL){
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    char type[3];
    fscanf(f, "%s", type);

    if (strcmp(type, "P3") != 0){
        fclose(f);
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fscanf(f, "%i", &width);
    fscanf(f, "%i", &height);

    int maxVal;
    fscanf(f, "%i", &maxVal);

    if (maxVal > 255){
        fclose(f);
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    for (int i = 0; i< width * height; i++){
        fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);
    }

    fclose(f);
}

void writePnm (const uchar3 *pixels, int width, int height, char *fileName){
	FILE *f = fopen(fileName, "w");

	if (f == NULL){
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "P3\n%i\n%i\n255\n", width, height);

	for (int i = 0; i < width * height; i++){
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	}

	fclose(f);
}

//Tạo bộ lọc
void initSobelFilter(int *filter, bool horizontal){

    int filterWidth = FILTER_WIDTH;
    int val = 0;
    int radius = filterWidth / 2;

    for (int filterR = 0; filterR < filterWidth; filterR++){

        for (int filterC = 0; filterC < filterWidth; filterC++){

        if (horizontal == true){
            if (filterC < radius){
            val = 1;
            }
            else if (filterC == radius){
            val = 0;
            }
            else{
            val = -1;
            }
            if (filterR == radius){
            val *= 2;
            }
        }
        else{
            if (filterR < radius){
            val = 1;
            }
            else if (filterR == radius){
            val = 0;
            }
            else{
            val = -1;
            }
            if (filterC == radius){
            val *= 2;
            }
        }

        filter[filterR * filterWidth + filterC] = val;
        }
    }
}

void convertRgbToGray (const uchar3 *inPixels, int n, int *outPixels){
    for (int i = 0; i < n; i++){
        outPixels[i] = 0.299f * inPixels[i].x + 0.587f * inPixels[i].y + 0.114f * inPixels[i].z;
    }
}

//-------------------------------------------------------Hàm trên host---------------------------------------------------------//
//Tính độ quan trọng của Pixels
void calPixelsImportance (int *inPixels, int width, int height, int *xFilter, int *yFilter, int filterWidth, int *outPixels){
    int radius = filterWidth / 2;
    for (int col = 0; col < width; col++){
        for (int row = 0; row < height; row++){

        int curIdx = row * width + col;
        float xSum = 0, ySum = 0;

        for (int filterRow = -radius; filterRow <= radius; filterRow++){
            for (int filterCol = -radius; filterCol <= radius; filterCol++){
            int filterIdx = (filterRow + radius) * filterWidth + filterCol + radius;

            int dx = min(width - 1, max(0, col + filterCol));
            int dy = min(height - 1, max(0, row + filterRow));

            int idx = dy * width + dx;
            xSum += inPixels[idx] * xFilter[filterIdx];
            ySum += inPixels[idx] * yFilter[filterIdx];
            }
        }

        outPixels[curIdx] = abs(xSum) + abs(ySum);
        }
    }
}

//Lấy pixels có độ quan trọng thấp nhất
void getLeastImportancePixels (int *inPixels, int width, int height, int *outPixels){
    int lastRow = (height - 1) * width;
    memcpy(outPixels + lastRow, inPixels + lastRow, width * sizeof(int));

    for (int row = height - 2; row >= 0; row--){
        int below = row + 1;

        for (int col = 0; col < width; col++ ){
        int idx = row * width + col;

        int leftCol = max(0, col - 1);
        int rightCol = min(width - 1, col + 1);

        int belowIdx = below * width + col;
        int leftBelowIdx = below * width + leftCol;
        int rightBelowIdx = below * width + rightCol;
        outPixels[idx] = min(outPixels[belowIdx], min(outPixels[leftBelowIdx], outPixels[rightBelowIdx])) + inPixels[idx];
        }
    }
}

//Lấy seam
void getSeam(int *inPixels, int width, int height, int *outPixels, int col){
    outPixels[0] = col;

    for (int row = 1; row < height; row++){
        int col = outPixels[row - 1];
        int idx = row * width + col;

        int leftCol = max(0, col - 1);
        int rightCol = min(width - 1, col + 1);

        int leftIdx = row * width + leftCol;
        int rightIdx = row * width + rightCol;

        if (inPixels[leftIdx] < inPixels[idx]){
        if (inPixels[leftIdx] < inPixels[rightIdx])
            outPixels[row] = leftCol;
        else
            outPixels[row] = rightCol;
        }
        else{
        if (inPixels[idx] < inPixels[rightIdx])
            outPixels[row] = col;
        else
            outPixels[row] = rightCol;
        }
    }
}

//Lấy seam có độ quan trọng thấp nhất
void getLeastImportanceSeam (int *inPixels, int width, int height, int *outPixels){
    int minCol = 0;
    for (int i = 0; i < width; i++){
        if (inPixels[i] < inPixels[minCol])
        minCol = i;
    }
    getSeam(inPixels, width, height, outPixels, minCol);
}

void removeSeam (const uchar3 *inPixels, int width, int height, uchar3 *outPixels, int *seam){
    int newWidth = width - 1;
    for (int row = 0; row < height; row++){
        int col = seam[row];
        memcpy(outPixels + row * newWidth, inPixels + row * width, col * sizeof(uchar3));

        int nextIdxOut = row * newWidth + col;
        int nextIdxIn = row * width + col + 1;
        memcpy(outPixels + nextIdxOut, inPixels + nextIdxIn, (newWidth - col) * sizeof(uchar3));
    }
}

void seamCarvingOnHost(const uchar3 *inPixels, int width, int height, uchar3 *outPixels, int *xFilter, int *yFilter, int filterWidth){
    //Chuyển ảnh về grayscale
    int *grayScalePixels = (int *)malloc(width * height * sizeof(int));
    convertRgbToGray(inPixels, width * height, grayScalePixels);

    //Edge detection
    int *pixelsImportance = (int *)malloc(width * height * sizeof(int));
    calPixelsImportance(grayScalePixels, width, height, xFilter, yFilter, filterWidth, pixelsImportance);

    //Tìm seam ít quan trọng nhất
    int *leastPixelsImportance = (int *)malloc(width * height * sizeof(int));
    getLeastImportancePixels(pixelsImportance, width, height, leastPixelsImportance);
    int *leastImportantSeam = (int *)malloc(height * sizeof(int));
    getLeastImportanceSeam(leastPixelsImportance, width, height, leastImportantSeam);

    //Xóa seam
    removeSeam(inPixels, width, height, outPixels, leastImportantSeam);

    //free memories
    free(grayScalePixels);
    free(pixelsImportance);
    free(leastPixelsImportance);
    free(leastImportantSeam);
}

//-----------------------------------------------------Hàm kernel------------------------------------------------------//
//v2 bổ sung song song hàm getLeastImportanceSeamKernel có sử dụng smem
__global__ void convertRgbToGrayKernel(uchar3 *inPixels, int width, int height, int *outPixels){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height){
        int idx = row * width + col;
        outPixels[idx] = 0.299f * inPixels[idx].x + 0.587f * inPixels[idx].y + 0.114f * inPixels[idx].z;
    }
}

__global__ void calPixelsImportanceKernel (int *inPixels, int width, int height, int filterWidth, int *outPixels){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height){
        int margin = filterWidth / 2;
        int curIdx = row * width + col;
        float xSum = 0, ySum = 0;

        for (int filterRow = -margin; filterRow <= margin; filterRow++){
        for (int filterCol = -margin; filterCol <= margin; filterCol++){
            int filterIdx = (filterRow + margin) * filterWidth + filterCol + margin;
            int dx = min(width - 1, max(0, col + filterCol));
            int dy = min(height - 1, max(0, row + filterRow));

            int idx = dy * width + dx;
            xSum += inPixels[idx] * dc_xFilter[filterIdx];
            ySum += inPixels[idx] * dc_yFilter[filterIdx];
        }
        }

        outPixels[curIdx] = abs(xSum) + abs(ySum);

    }
}

__global__ void getLeastImportancePixelsKernel (int *inPixels, int width, int row, int *outPixels){
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if (col < width){
        int idx = row * width + col;

        int below = row + 1;

        int leftCol = max(0, col - 1);
        int rightCol = min(width - 1, col + 1);

        int belowIdx = below * width + col;
        int leftBelowIdx = below * width + leftCol;
        int rightBelowIdx = below * width + rightCol;

        outPixels[idx] = min(outPixels[belowIdx], min(outPixels[leftBelowIdx], outPixels[rightBelowIdx])) + inPixels[idx];

    }
}

//Tìm seam có độ quan trọng nhỏ nhất trên device
__global__ void getLeastImportanceSeamKernel (int *inPixels, int width, int *outPixels){
    extern __shared__ int s_mem[];
    int i = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
    if (i < width)
      s_mem[threadIdx.x] = i;
    if (i + 1 < width)
      s_mem[threadIdx.x + 1] = i + 1;
    __syncthreads();
  
    for (int stride = 1; stride < 2 * blockDim.x; stride *= 2){
      if (threadIdx.x % stride == 0){
        if (i + stride < width){
          if (inPixels[s_mem[threadIdx.x]] > inPixels[s_mem[threadIdx.x + stride]]){
            s_mem[threadIdx.x] = s_mem[threadIdx.x + stride];
          }
        }
      }
      __syncthreads();
    }
  
    if (threadIdx.x == 0){
      outPixels[blockIdx.x] = s_mem[0];
    }
  }

void seamCarvingOnDevice(const uchar3 *inPixels, int width, int height, uchar3 *outPixels, int *xFilter, int *yFilter, int filterWidth, dim3 blockSize){
    int lastRowIdx = (height - 1) * width;
    int rowGridSize = (width - 1) / blockSize.x + 1;

    size_t dataSize = width * height * sizeof(uchar3);
    size_t rowSize = width * sizeof(int);
    size_t grayScaleSize = width * height * sizeof(int);
	int minColGridSize = (width - 1) / (2 * blockSize.x) + 1;
	
    dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

    // allocate device memories
    uchar3 *d_in;
	int *d_grayScalePixels, *d_pixelsImportance, *d_leastImportantPixels, *d_minCol;
	
    CHECK(cudaMalloc(&d_in, dataSize));
    CHECK(cudaMalloc(&d_grayScalePixels, grayScaleSize));
    CHECK(cudaMalloc(&d_pixelsImportance, grayScaleSize));
    CHECK(cudaMalloc(&d_leastImportantPixels, grayScaleSize));
    CHECK(cudaMalloc(&d_minCol, minColGridSize * sizeof(int)));
	
	int *leastPixelsImportance = (int *)malloc(grayScaleSize);
    int *leastImportantSeam = (int *)malloc(height * sizeof(int));
	int *minCol = (int *)malloc(minColGridSize * sizeof(int));

    //copy dữ liệu từ host vào device
    CHECK(cudaMemcpy(d_in, inPixels, dataSize, cudaMemcpyHostToDevice));

    //Chuyển hình ảnh sang grayscale
    convertRgbToGrayKernel<<<gridSize, blockSize>>>(d_in, width, height, d_grayScalePixels);
    CHECK(cudaGetLastError());

    //Thực hiện edge detection để lấy ra được độ quan trọng của pixels
    calPixelsImportanceKernel<<<gridSize, blockSize>>>(d_grayScalePixels, width, height, filterWidth, d_pixelsImportance);
    CHECK(cudaGetLastError());

    //Tìm pixels với độ quan trọng thấp nhất
    CHECK(cudaMemcpy(d_leastImportantPixels + lastRowIdx, d_pixelsImportance + lastRowIdx, rowSize, cudaMemcpyDeviceToDevice));
    for (int row = height - 2; row >= 0; row--){
        getLeastImportancePixelsKernel<<<rowGridSize, blockSize.x>>>(d_pixelsImportance, width, row, d_leastImportantPixels);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaMemcpy(leastPixelsImportance, d_leastImportantPixels, grayScaleSize, cudaMemcpyDeviceToHost));
    
    //Tìm seam có độ quan trọng thấp nhất
    //Chuyển thành song song từ v2
	getLeastImportanceSeamKernel<<<minColGridSize, blockSize.x, blockSize.x * 2 * sizeof(int)>>>(d_leastImportantPixels, width, d_minCol);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(minCol, d_minCol, minColGridSize * sizeof(int), cudaMemcpyDeviceToHost));
    int mc = minCol[0];
    for (int i = 0; i < minColGridSize; i += 1){
        if (leastPixelsImportance[minCol[i]] < leastPixelsImportance[mc]){
            mc = minCol[i];
        }
    }
	getSeam(leastPixelsImportance, width, height, leastImportantSeam, mc);
    //xóa seam 
    removeSeam(inPixels, width, height, outPixels, leastImportantSeam);

    //free memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_grayScalePixels));
    CHECK(cudaFree(d_pixelsImportance));
    CHECK(cudaFree(d_leastImportantPixels));
    free(leastPixelsImportance);
    free(leastImportantSeam);
}

void seamCarving(const uchar3 *inPixels, int width, int height, uchar3 *outPixels, int newWidth, int *xFilter, int *yFilter, int filterWidth, bool usingDevice=false, dim3 blockSize=dim3(1, 1)){
    if (usingDevice == false){
        printf("\nSeam carving by host\n");
    }
    else{
        printf("\nSeam carving by device\n");

        // copy x filter, y filter on host to dc_x filter, dc_y filter on device
        size_t filterSize = filterWidth * filterWidth * sizeof(int);
        CHECK(cudaMemcpyToSymbol(dc_xFilter, xFilter, filterSize));
        CHECK(cudaMemcpyToSymbol(dc_yFilter, yFilter, filterSize));
    }

    GpuTimer timer;
    timer.Start();

    //Khai báo biết temp để chứa dữ liệu trong quá trình seamCarving
    uchar3 *temp_in = (uchar3 *)malloc(width * height * sizeof(uchar3));
    uchar3 *temp_out = (uchar3 *)malloc(width * height * sizeof(uchar3));

    //Chuyển dữ liệu từ in vào temp_in
    memcpy(temp_in, inPixels, width * height * sizeof(uchar3));

    //Thự hiện remove seam có độ quan trọng thấp nhất đến khi đạt được chiều rộng mong muốn
    for (int w = width; w > newWidth; w--){
        //Chiều rộng của temp_out giảm đi 1 sau mỗi lần lặp
        temp_out = (uchar3 *)realloc(temp_out, (w-1) * height * sizeof(uchar3));

        //seamCarvingOnHost
        if (usingDevice == false){
            seamCarvingOnHost(temp_in, w, height, temp_out, xFilter, yFilter, filterWidth);
        }
        else{
            seamCarvingOnDevice(temp_in, w, height, temp_out, xFilter, yFilter, filterWidth, blockSize);
        }

        //Thực hiện swap dữ liệu temp_out cho temp_in để chuẩn bị cho vòng lặp sau
        uchar3 * temp = temp_in;
        temp_in = temp_out;
        temp_out = temp;
    }

    //Copy dữ liệu từ biến temp_in của vòng lặp cuối ra biến out để tiến hàng lưu file
    memcpy(outPixels, temp_in, newWidth * height * sizeof(uchar3));
  
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Eplapsed());
}

float computeError (uchar3 *a1, uchar3* a2, int n){
    float err = 0;
    for (int i = 0; i < n; i++){
        err += abs((int)a1[i].x - (int)a2[i].x);
        err += abs((int)a1[i].y - (int)a2[i].y);
        err += abs((int)a1[i].z - (int)a2[i].z);
    }
    err /= (n * 3);

    return err;
}

void printError (uchar3 *a1, uchar3 *a2, int width, int height){
    float err = computeError(a1, a2, width * height);
    printf("Error: %f\n", err);
}

void printDeviceInfo(){
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);
    printf("****************************\n");
}

char *concatStr(const char *s1, const char *s2){
    char *result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);

    return result;
}

int main (int argc, char **argv){
    if (argc != 3 && argc != 5){
        printf("Tham số không hợp lệ\n");
        return EXIT_FAILURE;
    }
	
	int width, height;
    uchar3 *inPixels;
	
    //Đọc file input
    readPnm(argv[1], width, height, inPixels);
    printf("\n Kích thước ảnh input (width * height): %i x %i\n", width, height);
	
    int newWidth = atoi(argv[2]);
    if (newWidth <= 0 || newWidth > width){
        printf("Kích thước ảnh mới không được nhỏ hơn 1 và lớn hơn kích thước file input");
        return EXIT_FAILURE;
    }
    printf("\n Kích thước ảnh mới (width * height): %i x %i\n", newWidth, height);

    printDeviceInfo();

    //Output & outputcorrect
	uchar3 *outPixels = (uchar3 *)malloc(newWidth * height * sizeof(uchar3));
    uchar3 *correctOutPixels = (uchar3 *)malloc(newWidth * height * sizeof(uchar3));
    

    //Cài đặt x-sobel filter & y-sobel filter
    int filterWidth = FILTER_WIDTH;
    int *xFilter = (int *)malloc(filterWidth * filterWidth * sizeof(int));
    int *yFilter = (int *)malloc(filterWidth * filterWidth * sizeof(int));
    initSobelFilter(xFilter, true);
    initSobelFilter(yFilter, false);

    //seamCarvingOnHost
    seamCarving(inPixels, width, height, correctOutPixels, newWidth, xFilter, yFilter, filterWidth);
    
    //Mặc định blockSize 32 x 32
    dim3 blockSize(32, 32);
    if (argc == 5){
        blockSize.x = atoi(argv[3]);
        blockSize.y = atoi(argv[4]);
    }

    //seamCarvingOnDevice
    seamCarving(inPixels, width, height, outPixels, newWidth, xFilter, yFilter, filterWidth, true, blockSize);
    printError(correctOutPixels, outPixels, newWidth, height);
   
    // Write results to file
	//Lấy tên file theo kích thước mới
    char *outFileNameBase = strtok(argv[2], ".");
    writePnm(correctOutPixels, newWidth, height, concatStr(outFileNameBase, "_host.pnm"));
    writePnm(outPixels, newWidth, height, concatStr(outFileNameBase, "_device.pnm"));

    // Free memories
    free(inPixels);
    free(xFilter);
    free(yFilter);
    free(correctOutPixels);
    free(outPixels);
}