#ifndef LU_ALGORITHM_HEADER
#define LU_ALGORITHM_HEADER

#define GSL //this will enable the GNU scientific library

#include "FunctionsLib.h"
#ifdef GSL
#include <gsl/gsl_multifit.h>
#endif
#include <iostream> 
#include <vector> 

using namespace std;

struct Coordinates {
  int x;
  int y;
};

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
IplImage * lu_algorithm(IplImage * image,int ks=1);
#endif