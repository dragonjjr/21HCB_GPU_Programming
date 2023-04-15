#Tích chập ảnh grayscale với bộ filter
void convolution(uint8_t *inPixels, int width, int height, int *outPixels, const int *filter) 
{
    for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
	{
		for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
		{
			int outPixel = 0;
			for (int filterR = 0; filterR < FILTER_WIDTH; filterR++)
			{
				for (int filterC = 0; filterC < FILTER_WIDTH; filterC++)
				{
					int filterVal = filter[filterR*FILTER_WIDTH + filterC];
					int inPixelsR = outPixelsR - FILTER_WIDTH/2 + filterR;
					int inPixelsC = outPixelsC - FILTER_WIDTH/2 + filterC;
					inPixelsR = min(max(0, inPixelsR), height - 1);
					inPixelsC = min(max(0, inPixelsC), width - 1);
					uint8_t inPixel = inPixels[inPixelsR*width + inPixelsC];
					outPixel += filterVal * inPixel;
				}
			}
			outPixels[outPixelsR*width + outPixelsC] = outPixel; 
		}
	}
}

void edgeDetection(uint8_t *inPixels, int width, int height, uint *importancePixels) 
{
    // Chuyển ảnh RGB sang ảnh grayscale: gray = 0.299*red + 0.587*green + 0.114*blue 
    uint8_t *grayPixels = (uint8_t*)malloc(width*height*sizeof(uint8_t));
    for (int i = 0; i < height; i++) 
    {
        for (int j = 0; j < width; j++)
        {
            int idx = width*i + j;
            grayPixels[idx] = 0.299f*inPixels[3*idx] + 0.587f*inPixels[3*idx + 1] + 0.114f*inPixels[3*idx + 2];
        }
    }

    // Phát hiện cạnh theo chiều x: Convolution với bộ lọc x-sobel
    int *edgePixels_x = (int*)malloc(width*height*sizeof(int));
    convolution(grayPixels, width, height, edgePixels_x, x_sobel_filter);

    // Phát hiện cạnh theo chiều y: Convolution với bộ lọc y-sobel
    int *edgePixels_y = (int*)malloc(width*height*sizeof(int));
    convolution(grayPixels, width, height, edgePixels_y, y_sobel_filter);

    // Tính độ quan trọng của một pixel
    for (int i = 0; i < width*height; i++)
        importancePixels[i] = abs(edgePixels_x[i]) + abs(edgePixels_y[i]);

    // Giải phóng vùng nhớ
    free(grayPixels);
    free(edgePixels_x);
    free(edgePixels_y);
}