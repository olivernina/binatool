#include "SuAlgorithm.h"

void get_normalization_parameters(IplImage * image,IplImage * min_image,IplImage * max_image, float& rmax_value,float& rmin_value){
	float max_value=0;
	float min_value=255;
	for(int y=0;y<image->height;y++){
		for(int x=0;x<image->width;x++){
			float value = min_max_operation(y,x,image,min_image,max_image);
			if(value > max_value){
				max_value = value;
			}
			else if(value < min_value){
				min_value = value;
			}
		}
	}
	rmax_value = max_value;
	rmin_value = min_value;
}

float min_max_operation(int y,int x,IplImage * image,IplImage * min_image,IplImage* max_image){

	int max_val = get_pixel(max_image,y,x);
	int min_val = get_pixel(min_image,y,x);
	int e = .001;
	float new_value = (float)(max_val - min_val)/(max_val + min_val + e);
	return new_value;
}


IplImage * min_max_edge_detection(IplImage * image){
	float overall_max;
	float overall_min;
	int kernel_size = 3;

	IplImage * max_image = cvCreateImage(cvGetSize(image),image->depth,image->nChannels); 
	IplImage * min_image = cvCreateImage(cvGetSize(image),image->depth,image->nChannels); 
	int values [] = {0,0,0,0,0,0,0,0,0};
	IplConvKernel * kernel = cvCreateStructuringElementEx(kernel_size,kernel_size,1,1,CV_SHAPE_RECT,values);
	cvErode(image,min_image,kernel);
	cvDilate(image,max_image,kernel);

	uchar* row_src,*row_max,*row_min,*row_dst;
	IplImage * dst_image = cvCreateImage(cvGetSize(image),image->depth,image->nChannels);
	for(int y=0;y<image->height;y++){
		row_src = get_row_data(image,y);
		row_max = get_row_data(max_image,y);
		row_min = get_row_data(min_image,y);
		row_dst = get_row_data(dst_image,y);

		for(int x=0;x<image->width;x++){
			float max_val = get_pixel_byte(row_max,x);
			float min_val = get_pixel_byte(row_min,x);
			int e = .001; //this is supposed to be an infinitely small and positive value

			//int new_value = 255 * enhanced_value/(overall_max-overall_min); //this might be more accurate but takes more time to compute it
			int new_val = ((max_val - min_val)/(max_val + min_val + e))*255; //At this line we are assuming that the \
			range of the resulting factions is from 0-1, However a more precise way would be to calculate the real max and min values\
			of the range and then do a normalization on this range, however, this would take another pass on the whole image which affects performance.
			set_pixel_byte(row_dst,x,new_val);
		}
	}
	cvReleaseImage(&max_image);
	cvReleaseImage(&min_image);
	return dst_image;

}

float get_edges_std(IplImage * edges_img,IplImage *image, float Emean,int i,int j,int radius){
	int kernel_width = i+ radius;
	int kernel_height =  j+ radius;
	float N= 0;
	float std_sum = 0;
	float Estd = 0;
	for( int x=i-radius;x< kernel_width+1;x++){
		for(int y=j-radius;y< kernel_height+1;y++){
			int edge_pixel = get_pixel(edges_img,y,x);
			if(edge_pixel == 0){
				int pixel = get_pixel(image,y,x);
				std_sum = pow((pixel-Emean),2) + std_sum;
				N++;
			}
		}
	}
	Estd = sqrt(std_sum/N);
	return Estd;
}

float get_edges_std(IplImage * edges_image,IplImage *src_image, float Emean,int i,int j,int radius,int edge_value){
	int kernel_width = i+ radius;
	int kernel_height =  j+ radius;
	float N= 0;
	float std_sum = 0;
	float Estd = 0;
	uchar* edges_row;
	uchar* src_row;
	for(int y=j-radius;y< kernel_height+1;y++){
		edges_row = get_row_data(edges_image,y);
		src_row = get_row_data(src_image,y);
		for( int x=i-radius;x< kernel_width+1;x++){
			int edge_pixel = get_pixel_byte(edges_row,x);
			if(edge_pixel == edge_value){
				int pixel = get_pixel_byte(src_row,x);
				std_sum = pow((pixel-Emean),2) + std_sum;
				N++;
			}
		}
	}
	Estd = sqrt(std_sum/N);
	return Estd;
}


IplImage* get_peak_pixels(IplImage* edge_image){
	
	IplImage* dst_image = cvCreateImage(cvGetSize(edge_image),edge_image->depth,edge_image->nChannels);
	cvZero(dst_image);
	int kernel_radius = 1;
	uchar* row_edge_img,*row_dst_img;
	uchar fore_val = 255;
	uchar back_val = 0;
	for(int y=0;y<edge_image->height;y++){
		row_edge_img = get_row_data(edge_image,y);
		row_dst_img = get_row_data(dst_image,y);
		for(int x=0;x<edge_image->width;x++){	
			uchar center_value = get_pixel_byte(row_edge_img,x);
			bool center_is_peak = false;
			if(center_value ==fore_val)
				if(is_coordinate_inside_boundaries(y,x-kernel_radius,edge_image)&&is_coordinate_inside_boundaries(y,x+kernel_radius,edge_image))
				{
					uchar left_pixel = get_pixel_byte(row_edge_img,x-kernel_radius);
					uchar right_pixel = get_pixel_byte(row_edge_img,x+kernel_radius);
					if(left_pixel == back_val || right_pixel== back_val)
						center_is_peak= true;
				}

			if(center_is_peak){
				set_pixel_byte(row_dst_img,x,fore_val);
			}
		}
	}
	return dst_image;
}

int find_next_edge_pixel(uchar*row,int index,int edge_value,int img_width){
    int next_index = index; //in case I don't find a next pixel stroke
	for(int i=index+1;i<img_width;i++){
		if(row[i]== edge_value){
            int next_index = i;
            return next_index;
		}
	}
    return next_index;
}

float get_edges_mean(IplImage * edgesImage,IplImage *srcImage,int i,int j,int radius, int edgeValue, float & Ne){
	int kernel_width = i+ radius;
	int kernel_height =  j+ radius;
	float Esum = 0;
	float Emean = 0;
	uchar* edgesRow;
	uchar* srcRow;
	for(int y=j-radius;y< kernel_height+1;y++){
		edgesRow = get_row_data(edgesImage,y);
		srcRow = get_row_data(srcImage,y);
		for( int x=i-radius;x< kernel_width+1;x++){
			int edgePixel = get_pixel_byte(edgesRow,x);
			if(edgePixel == edgeValue){
				int pixel = get_pixel_byte(srcRow,x);
				Esum = Esum + pixel;
				Ne++;
			}
		}
	}
	if(Ne > 0)
		Emean = Esum/Ne;
	return Emean;
}


IplImage* perform_local_thresholding(IplImage* image,IplImage* edges_image,int text_width){

    int window_size = text_width * 2; //This is part of tunning the parameters\
	according to the paper the window size should be a t least twice the width stroke
    if(window_size%2==0) // to make it odd
        window_size++;
    
    int radius = floor((float)window_size /2);
    
    int Nmin = window_size;
    IplImage * dst_image = cvCloneImage(image);
	int height_limit  = image->height-radius;
	int width_limit = image->width-radius;
	int edge_value = 255;
	uchar* dst_row;
	uchar* src_row;
	for(int j=0; j< image->height ;j++){ //change radius to 130 ie to debug windows size
		dst_row = get_row_data(dst_image,j);
		src_row = get_row_data(image,j);
		for(int i=0 ; i < image->width;i++){
			if(j <radius || i < radius || j>= height_limit ||i >= width_limit){ //This is to make whie the borders, we assume there is no text in the borders
				set_pixel_byte(dst_row,i,255);
					continue;
			}

            float Ne = 0;
            float Emean = get_edges_mean(edges_image,image,i,j,radius,edge_value,Ne);
			float Estd = get_edges_std(edges_image,image,Emean,i,j,radius,edge_value);
			int pixel = get_pixel_byte(src_row,i);
			if( Ne >= Nmin && pixel <= (Emean+Estd/2)){
				set_pixel_byte(dst_row,i,0);
			}
			else{
				set_pixel_byte(dst_row,i,255);
			}
		}
	}
    
    return dst_image;
}

IplImage* su_algorithm(IplImage* image){
	IplImage* edges_image = min_max_edge_detection(image);
	//cvShowImage("edges",edges_image);
	IplImage * contrast_image = otsu_algorithm(edges_image);
	//cvShowImage("contrast image",contrast_image);
	//contrast_image = invert_image(contrast_image);
	IplImage* peak_pixels_image = get_peak_pixels(contrast_image);
	//cvShowImage("peaks",peak_pixels_image);
	int text_width = text_width_approximation(peak_pixels_image);
	IplImage* rimage = perform_local_thresholding(image,contrast_image,text_width);
	//cvShowImage("rimage",rimage);
	return rimage;
}