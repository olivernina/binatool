#ifndef FUNCTIONS_LIB_HEADER
#define FUNCTIONS_LIB_HEADER

//#include <iostream>
//#define _CRTDBG_MAP_ALLOC
//#include <crtdbg.h>
//
//#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
//#define new DEBUG_NEW

//#define _CRTDBG_MAP_ALLOC
//#include <stdlib.h>
//#include <crtdbg.h>

#include "cv.h"
#include "cxcore.h"
#include "highgui.h" 
#include "LuAlgorithm.h"
#include "NinaAlgorithm.h"
#include "BilateralFilter.h"
#include "SuAlgorithm.h"
#include <vector> 
#include <algorithm>
#include <iostream> 
#include <stdlib.h>

using namespace std;
class imagelibrary{
public:
	static unsigned char * iplimage_to_bytearray(IplImage * image,int bytes_per_line=-1);
};
namespace cvt
{
	/*#define getPixelByte(rowData,col)(uchar)(rowData)[col];*/
	#define get_pixel_byte(rowData,col)(uchar)(rowData)[col]; //synonym
	//#define setPixelByte(rowData,col,newValue)((uchar*)(rowData))[col]= (uchar)newValue;
	#define set_pixel_byte(rowData,col,newValue)((uchar*)(rowData))[col]= (uchar)newValue; //synonym
	uchar* get_row_data(IplImage * image,int row); //synonym
	IplImage * add_black_pixels(IplImage* image, IplImage * timage);
	IplImage * apply_threshold(IplImage* image,int threshold, int type=-1);
	unsigned char * apply_threshold(unsigned char * image_data, int width, int height, int bytes_per_line, int threshold);
	IplImage* background_subtraction( IplImage* image ,IplImage* back_image);
	void batch_thresholding(char * directory,char * filenames[],char * tdirectory,char * tfilenames[], int num_files, IplImage * (*function)(IplImage*));
	IplImage* bilateral_filter(IplImage * image, int spacestdv,int rangestdv);
	IplImage * bytearray_to_iplimage(unsigned char * image,int width, int height,int bytes_per_line);
	int calculate_median(IplImage *image);
	int count_black_pixels(IplImage * image);
	void clear_histogram(int * histogram,int size);
	void clear_2d_bool_array(bool ** a,int heigh, int width);
	IplImage * compensate_contrast_variation(IplImage *image, IplImage * backimage);
	IplImage * compensate_contrast_interactive(IplImage *image, IplImage * backimage, int C); 
	IplImage * compensate_contrast_variation_normalized(IplImage *image, IplImage * backimage);
	IplImage * compensate_contrast_variation_modified(IplImage *image, IplImage * backimage, IplImage * border_image);
	IplImage * convert_pixels_to_white(IplImage* image,int threshold);
	IplImage * create_white_image(int width,int height,int channels);
	int * create_histogram(int size);
	IplImage * despeckle(IplImage* timage,IplImage * image,IplImage * back_img);
	IplImage * despeckle_ninas(IplImage* timage,IplImage * image,IplImage * back_img);
	IplImage * despeckle_ninas_old(IplImage* timage,IplImage * image,IplImage * back_img);
	IplImage * despeckle_threshold(IplImage* timage,IplImage * compensated_img,IplImage * back_img,int threshold);
	void display_image(char * winName,IplImage* image);
	int get_pixel(IplImage * image, int row,int col);
	unsigned char get_pixel(unsigned char * image_data,int bytes_per_line, int y, int x);
	int * get_histogram(IplImage * image);
	int * get_fast_histogram(unsigned char* image_data, int width, int height, int bytes_per_line);
	int * get_histogram_no_white(IplImage * image);
	int get_max_value(IplImage *image, IplImage * backimage,double C); //should be moved to lu's section
	int get_max_value(IplImage* image);
	int get_max_value(vector<int>values);
	int get_min_value(vector<int>values);
	float get_median(vector<int> list);
	IplImage* invert_image(IplImage * image);
	//unsigned char * iplimage_to_bytearray(IplImage * image);
	unsigned char * iplimage_to_bytearray(IplImage * image,int bytes_per_line=-1);
	bool is_coordinate_inside_boundaries(int y, int x,IplImage * image);
	IplImage * iterative_background_estimation(IplImage* image,int kernelRadius=21,int iterations=3);
	IplImage * kittler_algorithm(IplImage * image);
	int kittler_threshold_histogram(int *hist);
	
	IplImage * morphological_close(IplImage * binary_image,int size=15);
	IplImage * niblack_algorithm(IplImage * image , int radius,double K );
	IplImage * nina_algorithm(IplImage * image);
	unsigned char * otsu_algorithm(unsigned char * image_data,int width ,int height, int bytes_per_line);
	IplImage* otsu_algorithm(IplImage * image);
	IplImage * otsu_algorithm_modified(IplImage * image);
	IplImage * otsu_algorithm_modified_nocompensation(IplImage * image, IplImage * binary_image);
	int otsu_threshold_histogram(int *hist);
	IplImage * paint_lsi(unsigned char *  image_data,int width, int height,int bytes_per_line);
	void print_array(int * a,int size);
	void print_2d_bool_array(bool ** a,int height,int width);
	IplImage *  sauvola_algorithm(IplImage * image, int radius , double K, double R );
	void set_pixel(IplImage * image, int row,int col,int new_value);
	void set_pixel(unsigned char * image_data, int bytes_per_line,int y, int x, unsigned char new_value);
	
	
	//experimental 
	int fast_otsu_algorithm(int* histogram);
	unsigned char* compute_lsi_map(unsigned char* src, int width, int height, int bytesPerLine, int threshold);
	unsigned char * get_otsu_lsi_map(unsigned char* image_data, int width, int height,int bytes_per_line);
	unsigned char* get_lsi_map(IplImage * binary_image,int bytes_per_line);
	unsigned char *  lsi_algorithm_version5(unsigned char * image_data, int width,int height,int bytes_per_line);
	
}

using namespace cvt;

#endif