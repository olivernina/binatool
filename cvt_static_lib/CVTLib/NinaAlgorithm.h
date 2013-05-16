#ifndef NINA_ALGORITHM_HEADER
#define NINA_ALGORITHM_HEADER


class Reconstructor{
	public:
		bool ** visited;
		IplImage * final_image;
		IplImage * current_layer;
		Reconstructor(IplImage * image);
		IplImage* add_layer(IplImage * layer);
		bool pixel_leads_to_stroke(int i, int j);
		bool is_coord_inside_boundaries(int m, int n);
		bool is_same_pixel(int i,int j,int m,int n);
		bool pixel_touches_stroke(int i, int j);

};

IplImage * bilateral_using_template(IplImage * originalImage,int spacestdv,int rangestdv,IplImage *templateImg);
IplImage * get_pixels_on_template(IplImage* image,IplImage * timage);
IplImage* hdibco_algorithm(IplImage * image);
IplImage* otsu_iterations(IplImage * originalImage);
int otsu_thresholding_no_white(IplImage * image);
IplImage * otsuI(IplImage * oimage);
IplImage * otsuI_rec(IplImage * oimage);
IplImage * otsuI_improved(IplImage * oimage);
void otsu_iteration(IplImage * image,IplImage * &thresholded_img, IplImage * &left_over_image, int &threshold);
IplImage * rotsu_version4(IplImage* oimage,int iter);
IplImage *  rotsu_version5(IplImage* oimage,int iter);
IplImage*  rotsu_version3(IplImage * oimage,int &threshold);
IplImage * rotsu_iteration(IplImage* image, IplImage* leftover_image);
IplImage *  dicta_algorithm(IplImage * image);
IplImage * selecttive_bilateral(IplImage * image);
IplImage *  cvpr_algorithm_version0(IplImage * image);
IplImage * cvpr_algorithm_version2(IplImage * image);

#endif