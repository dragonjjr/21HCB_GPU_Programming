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

//Đọc file pnm
void readPnm (char *fileName, int &width, int &height, uchar3 *&pixels){
  FILE *f = fopen(fileName, "r");

  if (f == NULL){
    printf("Cannot read %s\n", fileName);
    exit(EXIT_FAILURE);
  }

  char type[3];
  fscanf(f, "%s", type);

  // Check the type of input img
  if (strcmp(type, "P3") != 0){
    fclose(f);
    printf("File input must be RGB %s\n", fileName);
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
//Ghi thành file pnm
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

//Đọc ma trận filter từ file
void writeMatrixTxt (int *pixels, int width, int height, char *fileName){
  FILE *f = fopen(fileName, "w");

  if (f == NULL){
    printf("Cannot write %s\n", fileName);
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < height; i++){
    for (int j = 0; j < width; j++){
      fprintf(f, "%d ", pixels[i * width + j]);
    }
    fprintf(f, "\n");
  }

  fclose(f);

}

//Khởi tạo bộ lọc
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

void convertRgb2Gray (const uchar3 *in, int n, int *out){
  for (int i = 0; i < n; i++){
    out[i] = 0.299f * in[i].x + 0.587f * in[i].y + 0.114f * in[i].z;
  }
}

//Tìm độ quan trọng của các pixels
void findPixelsImportant (int *in, int width, int height, int *xFilter, int *yFilter, int filterWidth, int *out){
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

//Tìm pixels có độ quan trọng thấp nhất
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

//Lấy ra seam tại vị trí
void getSeamAt (int *in, int width, int height, int *out, int col){
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

//Lấy ra seam có độ quạn trọng thấp nhất
void getLeastImportantSeam (int *in, int width, int height, int *out){
  int minCol = 0;
  for (int i = 0; i < width; i++){
    if (in[i] < in[minCol])
      minCol = i;
  }
  getSeamAt(in, width, height, out, minCol);
}

//Xóa bỏ seam
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

//Hàm seamcarving trên host
void seamCarvingHost(const uchar3 *in, int width, int height, uchar3 *out, int *xFilter, int *yFilter, int filterWidth){
  //Chuyển ảnh pnm thành grayscale
  int *grayScalePixels = (int *)malloc(width * height * sizeof(int));
  convertRgb2Gray(in, width * height, grayScalePixels);

  //Dùng thuật toán edge detection để tìm độ quan trọng của pixels
  int *pixelsImportance = (int *)malloc(width * height * sizeof(int));
  findPixelsImportant(grayScalePixels, width, height, xFilter, yFilter, filterWidth, pixelsImportance);

  //Tìm seam có độ quan trọng thấp nhất trong ảnh
  int *leastPixelsImportance = (int *)malloc(width * height * sizeof(int));
  getLeastImportantPixels(pixelsImportance, width, height, leastPixelsImportance);
  int *leastImportantSeam = (int *)malloc(height * sizeof(int));
  getLeastImportantSeam(leastPixelsImportance, width, height, leastImportantSeam);

  //Xóa bỏ seam có độ quan trọng thấp nhất
  removeSeam(in, width, height, out, leastImportantSeam);

  free(grayScalePixels);
  free(pixelsImportance);
  free(leastPixelsImportance);
  free(leastImportantSeam);
}

//Hàm chuyển đổi ảnh màu thành grayscale trên device
__global__ void convertRgb2GrayKernel(uchar3 *in, int width, int height, int *out){
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height){
    int idx = row * width + col;
    out[idx] = 0.299f * in[idx].x + 0.587f * in[idx].y + 0.114f * in[idx].z;
  }
}

//Hàm tìm độ quan trọng của pixels trên device
__global__ void findPixelsImportantKernel (int *in, int width, int height, int filterWidth, int *out){
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
        xSum += in[idx] * dc_xFilter[filterIdx];
        ySum += in[idx] * dc_yFilter[filterIdx];
      }
    }

    out[curIdx] = abs(xSum) + abs(ySum);

  }
}

//Hàm tìm pixels có độ quan trọng thấp nhất
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


__global__ void upTriangle (int *in, int width, int height, int yStart, int yStop, int baseWith, int *out){
  int xStart = baseWith * blockIdx.x * blockDim.x + threadIdx.x * baseWith;
  if (xStart < width){
    int xStop = xStart + baseWith - 1;

    for (int y = yStart; y >= yStop; y--){
      for (int x = xStart; x <= xStop; x++){
        int idx = y * width + x;

        if (x < width){

          if (y == height - 1){
            out[idx] = in[idx];
          }
          else{
            int below = (y + 1) * width + x;
            int left = (y + 1) * width + max(0, x - 1);
            int right = (y + 1) * width + min(width - 1, x + 1);
            
            out[idx] = in[idx] + min(out[below], min(out[left], out[right]));
          }
        }

      }
      xStart += 1;
      xStop -= 1;
    }
  }

}

__global__ void downTriangle (int *in, int width, int height, int yStart, int yStop, int baseWith, int *out){
  int xStop = baseWith * (threadIdx.x + blockDim.x * blockIdx.x);
  int xStart = xStop - 1;

  for (int y = yStart; y >= yStop; y--){
    for (int x = xStart; x <= xStop; x++){
      if (x >= 0 && x < width){

        int idx = y * width + x;
        int below = (y + 1) * width + x;
        int left = (y + 1) * width + max(0, x - 1);
        int right = (y + 1) * width + min(width - 1, x + 1);
         
        out[idx] = in[idx] + min(out[below], min(out[left], out[right]));
      }
    }
    xStart -= 1;
    xStop += 1;
  }
}


__device__ int bCount = 0;
volatile __device__ int bCount1 = 0;

__global__ void resetCount(){
  if (threadIdx.x == 0){
    bCount = 0;
    bCount1 = 0;
  }
}

__global__ void getMinColSeam1 (int *in, int width, int *out){
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
        if (in[s_mem[threadIdx.x]] > in[s_mem[threadIdx.x + stride]]){
          s_mem[threadIdx.x] = s_mem[threadIdx.x + stride];
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0){
    out[blockIdx.x] = s_mem[0];
  }
  //int i = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
  //for (int stride = 1; stride < 2 * blockDim.x; stride *= 2){
  //  if (threadIdx.x % stride == 0){
  //    if (i + stride < width){
  //      in[i] = min(in[i], in[i + stride]);
  //    }
  //  }
  //  __syncthreads();
  //}

  //if (threadIdx.x == 0){
  //  out[blockIdx.x] = in[blockDim.x * blockIdx.x * 2];
  //}
}

__global__ void getMinColSeam2 (int *in, int width, volatile int *out){
  extern __shared__ int s_mem[];
  __shared__ int bi;
  if (threadIdx.x == 0){
    bi = atomicAdd(&bCount, 1);
  }

  __syncthreads();

  int i = (blockDim.x * bi + threadIdx.x) * 2;
  if (i < width)
    s_mem[threadIdx.x] = i;
  if (i + 1 < width)
    s_mem[threadIdx.x + 1] = i + 1;
  __syncthreads();

  for (int stride = 1; stride < 2 * blockDim.x; stride *= 2){
    if (threadIdx.x % stride == 0){
      if (i + stride < width){
        if (in[s_mem[threadIdx.x]] > in[s_mem[threadIdx.x + stride]]){
          s_mem[threadIdx.x] = s_mem[threadIdx.x + stride];
        }
      }
    }
    __syncthreads();
  }

  //if (threadIdx.x == 0){
  //  out[bi] = s_mem[0];
  //}
  if (threadIdx.x == 0){
    out[bi] = s_mem[0];
    if (bi > 0){
      while(bCount1 < bi) {}
      if (in[out[bi]] < in[out[0]])
        out[0] = out[bi];
    }
    bCount1 += 1;
  }
}

__global__ void getMinColSeam3 (int *in, int width, volatile int *out){
  extern __shared__ int s_mem[];
  __shared__ int bi;
  if (threadIdx.x == 0){
    bi = atomicAdd(&bCount, 1);
  }

  __syncthreads();

  int i = (blockDim.x * bi + threadIdx.x) * 2;
  if (i < width){
    s_mem[threadIdx.x] = in[i];
    s_mem[threadIdx.x + 2 * blockDim.x] = i;
  }
    s_mem[threadIdx.x] = in[i];
  if (i + 1 < width)
    s_mem[threadIdx.x + 1] = in[i+1];
  __syncthreads();

  for (int stride = 1; stride < 2 * blockDim.x; stride *= 2){
    if (threadIdx.x % stride == 0){
      if (i + stride < width){
        if (in[s_mem[threadIdx.x]] > in[s_mem[threadIdx.x + stride]]){
          s_mem[threadIdx.x] = s_mem[threadIdx.x + stride];
        }
      }
    }
    __syncthreads();
  }

  //if (threadIdx.x == 0){
  //  out[bi] = s_mem[0];
  //}
  if (threadIdx.x == 0){
    out[bi] = s_mem[0];
    if (bi > 0){
      while(bCount1 < bi) {}
      if (in[out[bi]] < in[out[0]])
        out[0] = out[bi];
    }
    bCount1 += 1;
  }
}
__global__ void getMinColSeam (int *in, int width, volatile int *out){
  extern __shared__ int s_mem[]; //0 : blockDim.x*2-1 contain value of in array, blockDim.x*2 : blockDim.x*4-1 contain col idx
  __shared__ int bi;
  //if (threadIdx.x == 0){
  //  bi = atomicAdd(&bCount, 1);
  //}

  //__syncthreads();
  int i = blockDim.x * blockIdx.x * 2 + threadIdx.x;
  if (i < width){
    s_mem[threadIdx.x + blockDim.x * 2] = threadIdx.x;
    s_mem[threadIdx.x] = in[i];
    //out[i] = i;
  }
  if (i + blockDim.x < width){
    s_mem[threadIdx.x + blockDim.x + blockDim.x * 2] = threadIdx.x + blockDim.x;
    s_mem[threadIdx.x + blockDim.x] = in[i + blockDim.x];
    //out[i + blockDim.x] = i + blockDim.x;
  }
  __syncthreads();

  for (int stride = blockDim.x; stride >= 1; stride /= 2){
    if (threadIdx.x < stride){
      if (i + stride < width){
        //  if (in[out[i]] < in[out[i + stride]])
        //    out[i] = out[i];
        //  else
        //    out[i] = out[i + stride];
        //s_mem[threadIdx.x + blockDim.x * 2] = threadIdx.x;
        if (s_mem[s_mem[threadIdx.x + blockDim.x * 2]] > s_mem[s_mem[threadIdx.x + stride + blockDim.x * 2]])
          s_mem[threadIdx.x + blockDim.x * 2] = s_mem[threadIdx.x + stride + blockDim.x * 2];
      }
    }
    __syncthreads();
  }

  //if (out != NULL && threadIdx.x == 0){
  //  out[bi] = s_mem[blockDim.x * 2] + bi * blockDim.x * 2;
  //  if (bi > 0){
  //    while (bCount1 < bi){}
  //    if (in[out[bi]] < in[out[0]]){
  //      out[0] = out[bi];
  //    }
  //  }
  //  bCount1 += 1;
  //}
  if (threadIdx.x == 0){
    out[blockIdx.x] = s_mem[blockDim.x * 2] + blockDim.x * blockIdx.x * 2;
  }
}

//Hàm xóa seam trên device
__global__ void removeSeamKernel (uchar3 *in, int width, int height, uchar3 *out, int newWidth, int *seam){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.y + threadIdx.x;

  if (row < height && col < newWidth){
    if (col < seam[row]){
        out[row * newWidth + col].x = in[row * width + col].x;
        out[row * newWidth + col].y = in[row * width + col].y;
        out[row * newWidth + col].z = in[row * width + col].z;
      }
      else{
        out[row * newWidth + col].x = in[row * width + col + 1].x;
        out[row * newWidth + col].y = in[row * width + col + 1].y;
        out[row * newWidth + col].z = in[row * width + col + 1].z;
      }
  }
}

//Hàm seamcarving trên device
void seamCarvingDevice(const uchar3 *in, int width, int height, uchar3 *out, int *xFilter, int *yFilter, int filterWidth, dim3 blockSize, int baseWith){
  int newWidth = width - 1;
  int minColGridSize = (width - 1) / (2 * blockSize.x) + 1;

  size_t dataSize = width * height * sizeof(uchar3);
  size_t grayScaleSize = width * height * sizeof(int);
  size_t newDataSize = newWidth * height * sizeof(uchar3);

  dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
  dim3 newGridSize((newWidth - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

  //allocate device memories
  uchar3 *d_in, *d_out;
  int *d_grayScalePixels, *d_pixelsImportance, *d_leastImportantPixels, *d_minCol, *d_seam;
  CHECK(cudaMalloc(&d_in, dataSize));
  CHECK(cudaMalloc(&d_grayScalePixels, grayScaleSize));
  CHECK(cudaMalloc(&d_pixelsImportance, grayScaleSize));
  CHECK(cudaMalloc(&d_leastImportantPixels, grayScaleSize));
  CHECK(cudaMalloc(&d_minCol, minColGridSize * sizeof(int)));
  CHECK(cudaMalloc(&d_seam, height * sizeof(int)));
  CHECK(cudaMalloc(&d_out, newWidth * height * sizeof(uchar3)));

  // allocate host memories
  int *leastPixelsImportance = (int *)malloc(grayScaleSize);
  int *leastImportantSeam = (int *)malloc(height * sizeof(int));
  int *minCol = (int *)malloc(minColGridSize * sizeof(int));

  // copy data to device memories
  CHECK(cudaMemcpy(d_in, in, dataSize, cudaMemcpyHostToDevice));

  // convert image to grayscale
  convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_in, width, height, d_grayScalePixels);
  CHECK(cudaGetLastError());

  // edge detection
  findPixelsImportantKernel<<<gridSize, blockSize>>>(d_grayScalePixels, width, height, filterWidth, d_pixelsImportance);
  CHECK(cudaGetLastError());

  // find the least important pixels
  //CHECK(cudaMemcpy(d_leastImportantPixels + lastRowIdx, d_pixelsImportance + lastRowIdx, rowSize, cudaMemcpyDeviceToDevice));
  //for (int row = height - 2; row >= 0; row--){
  //  getLeastImportantPixelsKernel<<<rowGridSize, blockSize.x>>>(d_pixelsImportance, width, row, d_leastImportantPixels);
  //  CHECK(cudaGetLastError());
  //}
  //CHECK(cudaMemcpy(leastPixelsImportance, d_leastImportantPixels, grayScaleSize, cudaMemcpyDeviceToHost));
  //writeMatrixTxt(leastPixelsImportance, width, height, "least1.txt");
  
  int stripHeight = baseWith % 2 == 0 ? baseWith / 2 + 1 : (baseWith + 1) / 2 + 1;

  int gridSize1 = (width - 1) / (blockSize.x * baseWith) + 1;

  for (int y = height - 1; y >= 0; y -= stripHeight){
    int yStart = y;
    int yStop = max(0, yStart - stripHeight + 1);
    
    upTriangle<<<gridSize1, blockSize.x>>>(d_pixelsImportance, width, height, yStart, yStop, baseWith, d_leastImportantPixels);

    yStart = max(0, yStart - 1);
    yStop = max(0, yStart - stripHeight + 1);

    downTriangle<<<gridSize1 + 1, blockSize.x>>>(d_pixelsImportance, width, height, yStart, yStop, baseWith, d_leastImportantPixels);
  }



  CHECK(cudaMemcpy(leastPixelsImportance, d_leastImportantPixels, grayScaleSize, cudaMemcpyDeviceToHost));
  // find the least important seam
  // song song hoa van chua ra chinh xac hoan toan loi la 1.2xx
  resetCount<<<1, 1>>>();
  //getMinColSeam<<<minColGridSize, blockSize.x, blockSize.x * 4 * sizeof(int)>>>(d_leastImportantPixels, width, d_minCol);
  //getMinColSeam2<<<minColGridSize, blockSize.x, blockSize.x * 2 * sizeof(int)>>>(d_leastImportantPixels, width, d_minCol);
  //CHECK(cudaGetLastError());
  //CHECK(cudaMemcpy(minCol, d_minCol, minColGridSize * sizeof(int), cudaMemcpyDeviceToHost));
  //int mc = minCol[0];
  //for (int i = 0; i < minColGridSize; i += 1){
  //  if (leastPixelsImportance[minCol[i]] < leastPixelsImportance[mc]){
  //    mc = minCol[i];
  //  }
  //}
  //printf("min col device1 : %d\n", leastPixelsImportance[mc]);
  getMinColSeam2<<<minColGridSize, blockSize.x, blockSize.x * 2 * sizeof(int)>>>(d_leastImportantPixels, width, d_minCol);
  int mc;
  CHECK(cudaMemcpy(&mc, d_minCol, sizeof(int), cudaMemcpyDeviceToHost));
  //printf("%d-%d\n", mc, leastPixelsImportance[mc]);
  //CHECK(cudaMemcpy(minCol2, d_minCol2, minColGridSize * sizeof(int), cudaMemcpyDeviceToHost));
  //printf("min col device2: %d\n", leastPixelsImportance[minCol2[0]]);
  //free(minCol2);

  // co loi la 0.012974 do co 2 cot trung nhau gia tri min

  // getLeastImportantSeam(leastPixelsImportance, width, height, leastImportantSeam);
  getSeamAt(leastPixelsImportance, width, height, leastImportantSeam, mc);
  // remove the least important seam
  // ham song song cho nay tang thoi gian chay so voi chay tuan tu
  CHECK(cudaMemcpy(d_seam, leastImportantSeam, height * sizeof(int), cudaMemcpyHostToDevice));
  removeSeamKernel<<<newGridSize, blockSize>>>(d_in, width, height, d_out, newWidth, d_seam);
  CHECK(cudaMemcpy(out, d_out, newDataSize, cudaMemcpyDeviceToHost));
  //removeSeam(in, width, height, out, leastImportantSeam);

  // free memories
  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_grayScalePixels));
  CHECK(cudaFree(d_pixelsImportance));
  CHECK(cudaFree(d_leastImportantPixels));
  CHECK(cudaFree(d_minCol));
  CHECK(cudaFree(d_seam));
  CHECK(cudaFree(d_out));
  free(minCol);
  free(leastPixelsImportance);
  free(leastImportantSeam);
}

void seamCarving(const uchar3 *in, int width, int height, uchar3 *out, int newWidth, int *xFilter, int *yFilter, int filterWidth, bool usingDevice=false, dim3 blockSize=dim3(1, 1), int baseWith = 3){
  GpuTimer timer;
  timer.Start();

  if (usingDevice == false){
    printf("\nSeam carving by host\n");

    // allocate host memories
    uchar3 *src = (uchar3 *)malloc(width * height * sizeof(uchar3));
    uchar3 *dst = (uchar3 *)malloc(width * height * sizeof(uchar3));

    // store the pointer for freeing
    uchar3 *originalSrc = src;
    uchar3 *originalDst = dst;

    // copy input data to src pointer
    memcpy(src, in, width * height * sizeof(uchar3));

    // do the seam carving by decrease width by 1 until newWidth
    for (int w = width; w > newWidth; w--){
      // resize the dst pointer with current width - 1;
      dst = (uchar3 *)realloc(dst, (w-1) * height * sizeof(uchar3));

      // seamCarving the picture
      seamCarvingHost(src, w, height, dst, xFilter, yFilter, filterWidth);

      // swap src and dst
      uchar3 * temp = src;
      src = dst;
      dst = temp;
    }

    // copy the output data to the out pointer
    memcpy(out, src, newWidth * height * sizeof(uchar3));

    // free memories
    free(originalDst);
    free(originalSrc);
  }else{
    printf("\nSeam carving by device\n");

    // allocate device memories
    uchar3 *src = (uchar3 *)malloc(width * height * sizeof(uchar3));
    uchar3 *dst = (uchar3 *)malloc(width * height * sizeof(uchar3));

    // store the pointer for freeing
    uchar3 *originalSrc = src;
    uchar3 *originalDst = dst;

    // copy x filter, y filter on host to dc_x filter, dc_y filter on device
    size_t filterSize = filterWidth * filterWidth * sizeof(int);
    CHECK(cudaMemcpyToSymbol(dc_xFilter, xFilter, filterSize));
    CHECK(cudaMemcpyToSymbol(dc_yFilter, yFilter, filterSize));

    // copy input data to src pointer
    memcpy(src, in, width * height * sizeof(uchar3));

    // do the seam carving by decrease width by 1 until newWidth
    for (int w = width; w > newWidth; w--){
      // resize the dst pointer with current width - 1;
      dst = (uchar3 *)realloc(dst, (w-1) * height * sizeof(uchar3));

      // seamCarving the picture
      seamCarvingDevice(src, w, height, dst, xFilter, yFilter, filterWidth, blockSize, baseWith);

      // swap src and dst
      uchar3 * temp = src;
      src = dst;
      dst = temp;
    }

    // copy the output data to the out pointer
    memcpy(out, src, newWidth * height * sizeof(uchar3));

    // free memories
    free(originalDst);
    free(originalSrc);
  }

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

void printDeviceInfo(int codeVer){
  cudaDeviceProp devProv;
  CHECK(cudaGetDeviceProperties(&devProv, 0));
  printf("Vesrion of code: %d\n", codeVer);
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
  if (argc != 4 && argc != 6){
    printf("The number of arguments is invalid\n");
    return EXIT_FAILURE;
  }

  int newWidth = atoi(argv[2]);
  int baseWith = atoi(argv[3]);

  // Read input image file
  int width, height;
  uchar3 *inPixels;
  readPnm(argv[1], width, height, inPixels);
  printf("\nImage size (width * height): %i x %i\n", width, height);
  if (newWidth > width){
    printf("New Width of image must be equal or smaller than image's width");
    return EXIT_FAILURE;
  }

  // print device info
  int codeVer = 190;
  printDeviceInfo(codeVer);

  // init out pointer
  uchar3 *correctOutPixels = (uchar3 *)malloc(newWidth * height * sizeof(uchar3));
  uchar3 *outPixels = (uchar3 *)malloc(newWidth * height * sizeof(uchar3));

  // Set up x sobel filter and y sobel filter
  int filterWidth = FILTER_WIDTH;
  int *xFilter = (int *)malloc(filterWidth * filterWidth * sizeof(int));
  int *yFilter = (int *)malloc(filterWidth * filterWidth * sizeof(int));
  initSobelFilter(xFilter, true);
  initSobelFilter(yFilter, false);

  // Seam carving not using device
  seamCarving(inPixels, width, height, correctOutPixels, newWidth, xFilter, yFilter, filterWidth);

  // get input block size
  dim3 blockSize(32, 32); //default
  if (argc == 6){
    blockSize.x = atoi(argv[3]);
    blockSize.y = atoi(argv[4]);
  }

  // Seam carving using device
  seamCarving(inPixels, width, height, outPixels, newWidth, xFilter, yFilter, filterWidth, true, blockSize, baseWith);
  printError(correctOutPixels, outPixels, newWidth, height);

  // Write results to files
  char *outFileNameBase = strtok(argv[2], "."); //get rid of extension
  writePnm(outPixels, newWidth, height, concatStr(outFileNameBase, "_device.pnm"));
  writePnm(correctOutPixels, newWidth, height, concatStr(outFileNameBase, "_host.pnm"));

  // Free memories
  free(inPixels);
  free(xFilter);
  free(yFilter);
  free(correctOutPixels);
  free(outPixels);
}