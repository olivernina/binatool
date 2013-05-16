#ifndef SU_ALGORITHM_HEADER
#define SU_ALGORITHM_HEADER

#include "FunctionsLib.h"

void get_normalization_parameters(IplImage * image,IplImage * min_image,IplImage * max_image, float& rmax_value,float& rmin_value);
float min_max_operation(int y,int x,IplImage * image,IplImage * min_image,IplImage* max_image);
IplImage * min_max_edge_detection(IplImage * image);
float get_edges_std(IplImage * edges_img,IplImage *image, float Emean,int i,int j,int radius);
IplImage* get_peak_pixels(IplImage* edge_image);
IplImage* perform_local_thresholding(IplImage* image,IplImage* edges_image,int text_width);
IplImage* su_algorithm(IplImage* image);
int find_next_edge_pixel(uchar*row,int index,int edge_value,int img_width);
float get_edges_mean(IplImage * edgesImage,IplImage *srcImage,int i,int j,int radius, int edgeValue, float & Ne);
#endif