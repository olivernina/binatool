#ifndef NINA_ALGORITHM
#define NINA_ALGORITHM

#include <vector> 

#include "cv.h"
#include "cxcore.h"
#include "highgui.h" 
#include <gsl/gsl_multifit.h>

using namespace std;

struct Coordinates {
  int x;
  int y;
};

#define get_pixel_byte(rowData,col)(uchar)(rowData)[col]; //synonym
#define set_pixel_byte(rowData,col,newValue)((uchar*)(rowData))[col]= (uchar)newValue; //synonym

int * create_histogram(int size);
int * get_histogram(IplImage * image);
IplImage * otsu_algorithm(IplImage * image);
IplImage * kittler_algorithm(IplImage * image);
IplImage* invert_image(IplImage * image);
IplImage * min_max_edge_detection(IplImage * image);
void display_image(char * winName,IplImage* image);
uchar* get_row_data(IplImage * image,int row);
IplImage * despeckle_threshold(IplImage* timage,IplImage * compensated_img,IplImage * back_img,int threshold);
int otsu_threshold_histogram(int *hist);
IplImage * despeckle(IplImage* timage,IplImage * image,IplImage * back_img);
IplImage * compensate_contrast_variation(IplImage *image, IplImage * backimage);
IplImage * compensate_contrast_variation_normalized(IplImage *image, IplImage * backimage);
void get_distance_next_stroke(uchar* row,int * histogram, int hist_size,int img_width);
IplImage * apply_threshold(IplImage* image,int threshold, int type=-1);
int kittler_threshold_histogram(int *hist);
int otsu_algorithm(int* histogram);
int get_pixel(IplImage * image, int row,int col);
void set_pixel(IplImage * image, int row,int col, int new_value);
IplImage * apply_logical_operators(IplImage * dispekled_img);
IplImage * background_estimation(IplImage * image, int ks);
vector<int> calculate_fit_polynomial(vector<int>signal ,double * coeffs,int degree);
IplImage * edge_detection(IplImage * image);
vector<int> iterative_fitting(vector<int>signal,vector<Coordinates> sampled_signal, int iter_limit);
IplImage* local_thresholding(IplImage * compensate_img, IplImage * edges_img,int text_width);
int get_max_value_edge(IplImage * image);
void get_distance_next_stroke(vector<int> row,int* histogram);
int get_number_edge_pixels(IplImage* edges_img,int i,int j,int radius);
float get_edges_mean(IplImage * edges_img,IplImage *image,int i,int j,int radius);
vector<int> get_row_image(IplImage * image,int i);
double * polynomial_fitting(vector<Coordinates>signal,int degree);
bool polyfit(int obs, int degree,double *dx, double *dy, double *store); /* n, p */
void print_array(double * a,int size);
void print_vector(vector<int>y);
vector<Coordinates> sample_signal(vector<int> signal, int ks);
int text_width_approximation(IplImage * image);
IplImage * test_algorithm(IplImage * image,int ks=1);
IplImage * final_algorithm(IplImage * image,int ks=1);
#endif