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

void initSobelFilter(int *filter, bool horizontal){

    int filterWidth = FILTER_WIDTH;
    int val = 0;
    int margin = filterWidth / 2;

    for (int filterR = 0; filterR < filterWidth; filterR++){

        for (int filterC = 0; filterC < filterWidth; filterC++){

        if (horizontal == true){
            if (filterC < margin){
            val = 1;
            }
            else if (filterC == margin){
            val = 0;
            }
            else{
            val = -1;
            }
            if (filterR == margin){
            val *= 2;
            }
        }
        else{
            if (filterR < margin){
            val = 1;
            }
            else if (filterR == margin){
            val = 0;
            }
            else{
            val = -1;
            }
            if (filterC == margin){
            val *= 2;
            }
        }

        filter[filterR * filterWidth + filterC] = val;
        }
    }
}

void convertRgbToGray (const uchar3 *in, int n, int *out){
    for (int i = 0; i < n; i++){
        out[i] = 0.299f * in[i].x + 0.587f * in[i].y + 0.114f * in[i].z;
    }
}

//Tính độ quan trọng của Pixels
void getPixelsImportance (int *in, int width, int height, int *xFilter, int *yFilter, int filterWidth, int *out){
    int margin = filterWidth / 2;
    for (int col = 0; col < width; col++){
        for (int row = 0; row < height; row++){

        int curIdx = row * width + col;
        float xSum = 0, ySum = 0;

        for (int filterRow = -margin; filterRow <= margin; filterRow++){
            for (int filterCol = -margin; filterCol <= margin; filterCol++){
            int filterIdx = (filterRow + margin) * filterWidth + filterCol + margin;

            int dx = min(width - 1, max(0, col + filterCol));
            int dy = min(height - 1, max(0, row + filterRow));

            int idx = dy * width + dx;
            xSum += in[idx] * xFilter[filterIdx];
            ySum += in[idx] * yFilter[filterIdx];
            }
        }

        out[curIdx] = abs(xSum) + abs(ySum);
        }
    }
}

//Lấy pixels có độ quan trọng thấp nhất
void getLeastImportantPixels (int *in, int width, int height, int *out){
    int lastRow = (height - 1) * width;
    memcpy(out + lastRow, in + lastRow, width * sizeof(int));

    for (int row = height - 2; row >= 0; row--){
        int below = row + 1;

        for (int col = 0; col < width; col++ ){
        int idx = row * width + col;

        int leftCol = max(0, col - 1);
        int rightCol = min(width - 1, col + 1);

        int belowIdx = below * width + col;
        int leftBelowIdx = below * width + leftCol;
        int rightBelowIdx = below * width + rightCol;
        out[idx] = min(out[belowIdx], min(out[leftBelowIdx], out[rightBelowIdx])) + in[idx];
        }
    }
}

//Lấy seam
void getSeam(int *in, int width, int height, int *out, int col){
    out[0] = col;

    for (int row = 1; row < height; row++){
        int col = out[row - 1];
        int idx = row * width + col;

        int leftCol = max(0, col - 1);
        int rightCol = min(width - 1, col + 1);

        int leftIdx = row * width + leftCol;
        int rightIdx = row * width + rightCol;

        if (in[leftIdx] < in[idx]){
        if (in[leftIdx] < in[rightIdx])
            out[row] = leftCol;
        else
            out[row] = rightCol;
        }
        else{
        if (in[idx] < in[rightIdx])
            out[row] = col;
        else
            out[row] = rightCol;
        }
    }
}

//Lấy seam có độ quan trọng thấp nhất
void getLeastImportantSeam (int *in, int width, int height, int *out){
    int minCol = 0;
    for (int i = 0; i < width; i++){
        if (in[i] < in[minCol])
        minCol = i;
    }
    getSeam(in, width, height, out, minCol);
}

void removeSeam (const uchar3 *in, int width, int height, uchar3 *out, int *seam){
    int newWidth = width - 1;
    for (int row = 0; row < height; row++){
        int col = seam[row];
        memcpy(out + row * newWidth, in + row * width, col * sizeof(uchar3));

        int nextIdxOut = row * newWidth + col;
        int nextIdxIn = row * width + col + 1;
        memcpy(out + nextIdxOut, in + nextIdxIn, (newWidth - col) * sizeof(uchar3));
    }
}

void seamCarvingOnHost(const uchar3 *in, int width, int height, uchar3 *out, int *xFilter, int *yFilter, int filterWidth){
    //Chuyển ảnh về grayscale
    int *grayScalePixels = (int *)malloc(width * height * sizeof(int));
    convertRgbToGray(in, width * height, grayScalePixels);

    //Edge detection
    int *pixelsImportance = (int *)malloc(width * height * sizeof(int));
    getPixelsImportance(grayScalePixels, width, height, xFilter, yFilter, filterWidth, pixelsImportance);

    //Tìm seam ít quan trọng nhất
    int *leastPixelsImportance = (int *)malloc(width * height * sizeof(int));
    getLeastImportantPixels(pixelsImportance, width, height, leastPixelsImportance);
    int *leastImportantSeam = (int *)malloc(height * sizeof(int));
    getLeastImportantSeam(leastPixelsImportance, width, height, leastImportantSeam);

    //Xóa seam
    removeSeam(in, width, height, out, leastImportantSeam);

    //free memories
    free(grayScalePixels);
    free(pixelsImportance);
    free(leastPixelsImportance);
    free(leastImportantSeam);
}

__global__ void convertRgbToGrayKernel(uchar3 *in, int width, int height, int *out){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height){
        int idx = row * width + col;
        out[idx] = 0.299f * in[idx].x + 0.587f * in[idx].y + 0.114f * in[idx].z;
    }
}

__global__ void getPixelsImportanceKernel (int *in, int width, int height, int filterWidth, int *out, int *xFilter, int *yFilter){
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
            xSum += in[idx] * xFilter[filterIdx];
            ySum += in[idx] * yFilter[filterIdx];
        }
        }

        out[curIdx] = abs(xSum) + abs(ySum);

    }
}

__global__ void getLeastImportantPixelsKernel (int *in, int width, int row, int *out){
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if (col < width){
        int idx = row * width + col;

        int below = row + 1;

        int leftCol = max(0, col - 1);
        int rightCol = min(width - 1, col + 1);

        int belowIdx = below * width + col;
        int leftBelowIdx = below * width + leftCol;
        int rightBelowIdx = below * width + rightCol;

        out[idx] = min(out[belowIdx], min(out[leftBelowIdx], out[rightBelowIdx])) + in[idx];

    }
}

void seamCarvingOnDevice(const uchar3 *in, int width, int height, uchar3 *out, int *xFilter, int *yFilter, int filterWidth, dim3 blockSize){
    int lastRowIdx = (height - 1) * width;
    int rowGridSize = (width - 1) / blockSize.x + 1;

    size_t dataSize = width * height * sizeof(uchar3);
    size_t rowSize = width * sizeof(int);
    size_t grayScaleSize = width * height * sizeof(int);

    dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

    // allocate device memories
    uchar3 *d_in;
    size_t filterSize = filterWidth * filterWidth * sizeof(int);
	  int * d_xFilter;
	  int * d_yFilter;
    int *d_grayScalePixels, *d_pixelsImportance, *d_leastImportantPixels;
    CHECK(cudaMalloc(&d_in, dataSize));
    CHECK(cudaMalloc(&d_xFilter, filterSize));
    CHECK(cudaMalloc(&d_yFilter, filterSize));
    CHECK(cudaMalloc(&d_grayScalePixels, grayScaleSize));
    CHECK(cudaMalloc(&d_pixelsImportance, grayScaleSize));
    CHECK(cudaMalloc(&d_leastImportantPixels, grayScaleSize));
	
	int *leastPixelsImportance = (int *)malloc(grayScaleSize);
    int *leastImportantSeam = (int *)malloc(height * sizeof(int));

    //copy dữ liệu từ host vào device
    CHECK(cudaMemcpy(d_in, in, dataSize, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_xFilter, xFilter, filterSize, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_yFilter, yFilter, filterSize, cudaMemcpyHostToDevice));

    //Chuyển hình ảnh sang grayscale
    convertRgbToGrayKernel<<<gridSize, blockSize>>>(d_in, width, height, d_grayScalePixels);
    CHECK(cudaGetLastError());

    //Thực hiện edge detection để lấy ra được độ quan trọng của pixels
    getPixelsImportanceKernel<<<gridSize, blockSize>>>(d_grayScalePixels, width, height, filterWidth, d_pixelsImportance, d_xFilter, d_yFilter);
    CHECK(cudaGetLastError());

    //Tìm pixels với độ quan trọng thấp nhất
    CHECK(cudaMemcpy(d_leastImportantPixels + lastRowIdx, d_pixelsImportance + lastRowIdx, rowSize, cudaMemcpyDeviceToDevice));
    for (int row = height - 2; row >= 0; row--){
        getLeastImportantPixelsKernel<<<rowGridSize, blockSize.x>>>(d_pixelsImportance, width, row, d_leastImportantPixels);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaMemcpy(leastPixelsImportance, d_leastImportantPixels, grayScaleSize, cudaMemcpyDeviceToHost));
    
    //Tìm seam có độ quan trọng thấp nhất
    getLeastImportantSeam(leastPixelsImportance, width, height, leastImportantSeam);

    //xóa seam 
    removeSeam(in, width, height, out, leastImportantSeam);

    //free memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_grayScalePixels));
    CHECK(cudaFree(d_pixelsImportance));
    CHECK(cudaFree(d_leastImportantPixels));
    free(leastPixelsImportance);
    free(leastImportantSeam);
}

void seamCarving(const uchar3 *in, int width, int height, uchar3 *out, int newWidth, int *xFilter, int *yFilter, int filterWidth, bool usingDevice=false, dim3 blockSize=dim3(1, 1)){
    if (usingDevice == false){
        printf("\nSeam carving by host\n");
    }
    else{
        printf("\nSeam carving by device\n");
    }

    GpuTimer timer;
    timer.Start();

    //Khai báo biết temp để chứa dữ liệu trong quá trình seamCarving
    uchar3 *temp_in = (uchar3 *)malloc(width * height * sizeof(uchar3));
    uchar3 *temp_out = (uchar3 *)malloc(width * height * sizeof(uchar3));

    //Chuyển dữ liệu từ in vào temp_in
    memcpy(temp_in, in, width * height * sizeof(uchar3));

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
    memcpy(out, temp_in, newWidth * height * sizeof(uchar3));
  
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
    char *outFileNameBase = strtok(argv[1], ".");
    writePnm(correctOutPixels, newWidth, height, concatStr(outFileNameBase, "_host.pnm"));
    writePnm(outPixels, newWidth, height, concatStr(outFileNameBase, "_device.pnm"));

    // Free memories
    free(inPixels);
    free(xFilter);
    free(yFilter);
    free(correctOutPixels);
    free(outPixels);
}