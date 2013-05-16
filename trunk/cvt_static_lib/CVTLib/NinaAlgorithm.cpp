#include "FunctionsLib.h"
#include "NinaAlgorithm.h"

using namespace cvt;

Reconstructor::Reconstructor(IplImage * image){
	this->final_image = cvCloneImage(image);
	visited = new bool *[image->height];
	for(int i=0;i<image->height;i++){
		visited[i] = new bool[image->width];
	}
	clear_2d_bool_array(visited, image->height,image->width);
	//print_2d_bool_array(visited,image->height,image->width);
}
IplImage * Reconstructor::add_layer(IplImage * layer){
	this->current_layer = cvCloneImage(layer);
	for(int i=0;i<layer->height;i++){
		for(int j=0;j<layer->width;j++){
			int pixel = get_pixel(layer,i,j);
			if(pixel==0){
				if(this->pixel_leads_to_stroke(i,j)){
					cvSet2D(this->final_image,i,j,cvScalar(0,0,0));
				}
				clear_2d_bool_array(this->visited, layer->height,layer->width);
			}
		}
	}
	return this->final_image;
}

bool Reconstructor::is_coord_inside_boundaries(int m, int n){
	if(m>-1 && n>-1 && m< this->final_image->height && n < this->final_image->width)
		return true;
    else 
        return false;
}

bool Reconstructor::is_same_pixel(int i,int j,int m,int n){
	if(i==m && j==n)
		return true;
	else
		return false;
}

bool Reconstructor::pixel_touches_stroke(int i, int j){
	for(int m = i -1; m <= i+1;m++){
		for( int n =j-1 ; n <= j+1 ; n++){
			if(this->is_coord_inside_boundaries(m, n) && !this->is_same_pixel(i,j,m,n)){
				int neighbor_pixel = get_pixel(this->final_image,m,n);
				if(neighbor_pixel==0)
					return true;
			}
		}
	}
	return false;
}


bool Reconstructor::pixel_leads_to_stroke(int i, int j){
	if(this->pixel_touches_stroke(i,j))
		return true;
	else{
		this->visited[i][j] = true;
		for(int m=i-1; m<=i+1; m++){
			for(int n=j-1; n<=j+1; n++){
				if(this->is_coord_inside_boundaries(m, n) && !this->is_same_pixel(i,j,m,n) && !this->visited[m][n]){
						this->visited[m][n]=true;
						int neighbor_node = get_pixel(this->current_layer,m,n);
						if(neighbor_node == 0){
							return this->pixel_leads_to_stroke(m,n);

						}
				}
			}
		}
		return false;
	}

}

int get_number_thresolded_pixels(IplImage* timg,int i,int j,int radius){
	int kernel_width = i+ radius;
	int kernel_height =  j+ radius;
	int Ne = 0;
	for(int x =i-radius;x < kernel_width+1;x++){
		for(int y=j-radius;y<kernel_height+1;y++){
			int pixel = get_pixel(timg,y,x);
			if(pixel ==0)
				Ne++;
		}
	}
	return Ne;
}

 int get_thresholded_pixels_mean(IplImage * timg,IplImage *img,int i,int j,int radius){
    int kernel_width = i+ radius;
    int kernel_height =  j+ radius;
    float Ne = 0;
    float Esum = 0;
    int Emean = 0;
		for( int x=i-radius;x< kernel_width+1;x++){
			for(int y=j-radius;y< kernel_height+1;y++){
				int tpixel = get_pixel(timg,y,x);
				if(tpixel == 0){
					int pixel = get_pixel(img,y,x);
					Esum = Esum + pixel;
					Ne++;
				}
			}
		}
    if(Ne > 0)
        Emean = ceil(Esum/Ne);
    return Emean;
}

 IplImage * get_left_over_pixels(IplImage* image,IplImage * timage){
	 IplImage * rimage = cvCloneImage(image);
	 for(int y = 0; y < image->height; y++){
		 for(int x = 0; x < image->width; x++){
			 int tpixel =get_pixel(timage,y,x);
			 if(tpixel==0)
				 cvSet2D(rimage,y,x,cvScalar(255,255,255));

			}
	 }
	 return rimage;
 }

IplImage * get_pixels_on_template(IplImage* image,IplImage * timage){
	 IplImage * rimage = cvCloneImage(image);
	 for(int y = 0; y < image->height; y++){
		 for(int x = 0; x < image->width; x++){
			 int tpixel =get_pixel(timage,y,x);
			 if(tpixel!=0)
				 cvSet2D(rimage,y,x,cvScalar(255,255,255));

			}
	 }
	 return rimage;
 }

int otsu_thresholding_no_white(IplImage * image){
    int * hist = get_histogram_no_white(image);
	//print_array(hist,256);
    int T = otsu_threshold_histogram(hist);
    return T;
}

void otsu_iteration(IplImage * image,IplImage * &thresholded_img, IplImage * &left_over_image, int &threshold){
    threshold = otsu_thresholding_no_white(image);
    printf("T: %d\n",threshold);
    thresholded_img = apply_threshold(image,threshold,-1);
    left_over_image= get_left_over_pixels(image,thresholded_img);
}

IplImage*  rotsu_version3(IplImage * oimage,int &threshold){
    IplImage * leftpixels = cvCloneImage(oimage);
	IplImage * thresholded_img = cvCloneImage(oimage);
	otsu_iteration(oimage,thresholded_img, leftpixels,threshold);
	//printf("T:%d\n",threshold);
    //display_image("first threhold",thresholded_img);
    //display_image("left pixels",leftpixels);
    IplImage * final_image = cvCloneImage(thresholded_img);
	int original_count = count_black_pixels(thresholded_img);
	int previous_count = 0;
    int current_count = 1;
    int counter = 1;
    int total_count = count_black_pixels(thresholded_img);
    int previous_threshold = -1;
	while(current_count <= original_count && current_count > 0) {
        previous_count = current_count;
        previous_threshold = threshold;
        otsu_iteration(leftpixels,thresholded_img,leftpixels,threshold);
        //display("left pixels"+str(counter),leftpixels);
        //display("t"+str(counter),thresholdedImg)
		//display_image("last threhold",thresholded_img);
        current_count = count_black_pixels(thresholded_img);
        int diff_threshold = threshold - previous_threshold;
		if(current_count <= original_count && current_count > 0 && diff_threshold >= 2 && diff_threshold < 27){
			final_image = add_black_pixels(final_image,thresholded_img);
		}
		else{
			break;
		}

        counter++;
        
	}
    return final_image;
}


IplImage *  rotsu_version4(IplImage* oimage,int iter){
    int threshold = -1;
	IplImage * timage;
	
	for(int i=0; i< iter;i++){
       timage= rotsu_version3(oimage,threshold);
        printf("threshold: %d\n",threshold);
        display_image("timg",timage);
        oimage = convert_pixels_to_white(oimage,threshold);
        display_image("ri",oimage);
	}

    return timage;
}
IplImage *  rotsu_version5(IplImage* oimage,int iter){
    int threshold = -1;
	IplImage * timage = cvCloneImage(oimage);
	IplImage * leftpixels = cvCloneImage(oimage);
	IplImage * thresholded_img = cvCloneImage(oimage);

	for(int i=0; i< iter;i++){
       timage= rotsu_version3(oimage,threshold);
        printf("threshold: %d\n",threshold);
        display_image("timg",timage);
        oimage = convert_pixels_to_white(oimage,threshold);
        display_image("ri",oimage);
	}
	otsu_iteration(oimage,thresholded_img, leftpixels,threshold);
	display_image("oimage",oimage);
	display_image("leftpixels",leftpixels);
	printf("thresh: %d",threshold);
    return thresholded_img;
}

void convert_pixels(IplImage* origImg,IplImage *threshImg,IplImage *destImg,int from,int to){ 
	unsigned char *imgData= reinterpret_cast<unsigned char *>(origImg->imageData);
	unsigned char *destData= reinterpret_cast<unsigned char *>(destImg->imageData);
	unsigned char *threshData= reinterpret_cast<unsigned char *>(threshImg->imageData);

	int h= origImg->height;
	int w= origImg->width;
	int step= origImg->widthStep/sizeof(uchar); 

	for(int y = 0; y < h; ++y){
	  for(int x = 0; x < w; ++x){
			destData[y*step+x] = (threshData[y*step+x] == from)? to:imgData[y*step+x];
			
		}
	}
}

IplImage * rotsu_iteration(IplImage* image, IplImage* leftover_image){
	int * histogram = get_histogram_no_white(image);
	int threshold = otsu_threshold_histogram(histogram);
	IplImage * thresholded_img = apply_threshold(image,threshold,-1);
	printf("Threshold: %d",threshold);
	convert_pixels(image,thresholded_img,leftover_image,0,255); 
	return thresholded_img;
}

IplImage * bilateral_using_template(IplImage * originalImage,int spacestdv,int rangestdv,IplImage *templateImg){ //Template must be white pixels 255
	BilateralFilter *bilateral = new BilateralFilter(originalImage,spacestdv,rangestdv,false);
	IplImage * rimage = cvCreateImage(cvGetSize(originalImage),originalImage->depth,originalImage->nChannels);

	for(int i=0;i<originalImage->height;i++){
		for(int j=0;j<originalImage->width;j++){

			CvScalar pixel = cvGet2D(templateImg,i,j);
			if(pixel.val[0]==255){
				bilateral ->apply(i,j);
			}
		}
	}
	
	rimage = bilateral->rimage;

	return rimage;
}

int otsu_with_template(IplImage *origImg,IplImage *Timg){
// this function takes an image an a template which is defined by black pixels and then calculates the otsu threshold based on the pixels on the original image that aren't in thes same position as the template
	unsigned char *origImgData= reinterpret_cast<unsigned char *>(origImg->imageData);
	unsigned char *TimgData= reinterpret_cast<unsigned char *>(Timg->imageData);
	int h= origImg->height;
	int w= origImg->width;
	int step= origImg->widthStep/sizeof(uchar);

	int * hist= create_histogram(256);
	
	for(int i=0;i<h;i++){
		for(int j=0;j<w;j++){
			int curPixel = origImgData[i*step+j];
			if(TimgData[i*step+j]>0x00){
				++hist[curPixel];
			}
		}
	}
	return otsu_threshold_histogram(hist);
}

IplImage* otsu_iterations(IplImage * originalImage){
	//This function does multiple iterations of the Otsu algorithm based on how many pixels we add to the final image. 
	//We keep iterating on Otsu until we exceed the number of pixels added to the final image to that of the previous iteration. This mean we are likekly adding pixels from the noisy background

	int currentCount = 0;
	IplImage *finalImage=cvCloneImage(originalImage);
	int * hist = get_histogram(originalImage);
	int threshold = otsu_threshold_histogram(hist);

	finalImage = apply_threshold(originalImage,threshold);
	
	int originalCount = count_black_pixels(finalImage);
	
	int counter=0;
	
	while(currentCount<=originalCount){
	
		IplImage * thresholdedImage  =cvCloneImage(originalImage);
		threshold = otsu_with_template(originalImage,finalImage);
		thresholdedImage = apply_threshold(originalImage,threshold);
		currentCount = count_black_pixels(thresholdedImage);
		if(currentCount<=originalCount){
			add_black_pixels(finalImage, thresholdedImage);
			
		}
		counter++;
	}
	printf("iterations:%d",counter);
	
	return finalImage;
}
IplImage * otsuI(IplImage * oimage){
	IplImage * leftpixels = cvCloneImage(oimage);
	IplImage * timage = rotsu_iteration(oimage,leftpixels);
	IplImage * final_image = cvCloneImage(timage);
	display_image("t",timage);
	cvSaveImage("output/firstlayer.png",timage);
	int current_count = 0;
	int original_count = count_black_pixels(timage);
	bool first_time = true;
	while(current_count <= original_count ){
		timage = rotsu_iteration(leftpixels,leftpixels);
		if(first_time){
			cvSaveImage("output/secondlayer.png",timage);
			first_time = false;
		}
		current_count = count_black_pixels(timage);
		if(current_count <= original_count ){
			final_image = add_black_pixels(final_image,timage);
		}

	}
	return final_image;
}

IplImage * otsuI_improved(IplImage * oimage){
	IplImage * leftpixels = cvCloneImage(oimage);
	int threshold = -1;
	IplImage * timage = cvCloneImage(oimage);
	otsu_iteration(oimage,timage,leftpixels,threshold);

	Reconstructor reconstructor(timage);
	IplImage * final_image = cvCloneImage(timage);
	display_image("t",timage);
	cvSaveImage("output/firstlayer.png",timage);
	int current_count = 0;
	int original_count = count_black_pixels(timage);
	int previous_threshold = -1;
	
	bool first_time = true;
	IplImage * secondlayer;
	bool fill_holes= false;
	while(current_count <= original_count){
		previous_threshold = threshold;
		otsu_iteration(leftpixels,timage,leftpixels,threshold);
		
		current_count = count_black_pixels(timage);
		int diff_threshold = threshold - previous_threshold;
		
		if(first_time){
			cvSaveImage("output/secondlayer.png",timage);
			secondlayer = cvCloneImage(timage);
			first_time = false;
		}

		if(current_count <= original_count && current_count > 0 && diff_threshold >= 2 && diff_threshold < 30 && threshold < 249){
			final_image = reconstructor.add_layer(timage);
			fill_holes = true;
		}
		else{
			break;
		}
	}
	if(fill_holes)
		final_image = reconstructor.add_layer(secondlayer);
	return final_image;
}
IplImage * otsuI_rec(IplImage * oimage){
	IplImage * leftpixels = cvCloneImage(oimage);
	IplImage * timage = rotsu_iteration(oimage,leftpixels);
	Reconstructor reconstructor(timage);
	IplImage * final_image = cvCloneImage(timage);
	display_image("t",timage);
	
	int current_count = 0;
	int original_count = count_black_pixels(timage);
	bool first_time = true;
	IplImage * secondlayer;
	while(current_count <= original_count){
		timage = rotsu_iteration(leftpixels,leftpixels);
		if(first_time){
			secondlayer = cvCloneImage(timage);
			first_time = false;
		}
		current_count = count_black_pixels(timage);
		if(current_count <= original_count ){
			final_image = reconstructor.add_layer(timage);
			
		}
	}
	final_image = reconstructor.add_layer(secondlayer);
	return final_image;
}

IplImage* hdibco_algorithm(IplImage * image){
	IplImage *img_background = iterative_background_estimation(image,21);
    IplImage * compensated_img = compensate_contrast_variation_normalized(image,img_background);
    int range_parameter = 2;
    int space_parameter =10;
    IplImage * bilateralImg = bilateral_filter(compensated_img, space_parameter, range_parameter);
	int threshold = -1;
	IplImage * finalImage = rotsu_version3(bilateralImg,threshold);
	finalImage = despeckle_ninas(finalImage,image,img_background);
    return finalImage;
}

IplImage *  dicta_algorithm(IplImage * image){
	IplImage * img_background = cvCloneImage(image);
	cvSmooth(image,img_background,3,21,21);
	IplImage * img_wo_back = background_subtraction(image,img_background);
	img_wo_back = invert_image(img_wo_back);
	IplImage * bilateral_img = bilateral_filter(img_wo_back,10,2);
	IplImage * timage = otsuI(bilateral_img);
    IplImage * back_smoothed = bilateral_using_template(bilateral_img, 10, 3, timage);
	timage = otsuI(back_smoothed);
    timage = invert_image(timage);
    IplImage * fore_smoothed = bilateral_using_template(back_smoothed, 2, 2, timage);
	timage = otsuI_improved(fore_smoothed);
	display_image("final image",timage);
	return timage;

}

IplImage * selecttive_bilateral(IplImage * image){
	IplImage * timage = otsuI(image);
    IplImage * back_smoothed = bilateral_using_template(image, 10, 3, timage);
	display_image("Background smoothing", back_smoothed);
	timage = otsuI(back_smoothed);
    display_image("Foreground smoothing template I", timage);
    timage = invert_image(timage);
    IplImage * fore_smoothed = bilateral_using_template(back_smoothed, 2, 2, timage);
	return fore_smoothed;
}

IplImage *  cvpr_algorithm_version0(IplImage * image){
	IplImage * back_img = iterative_background_estimation(image,41,1);
	IplImage *compensated_img = compensate_contrast_variation(image,back_img);
	IplImage * otsu_image = otsu_algorithm(compensated_img);
	return otsu_image;
}
IplImage * cvpr_algorithm_version2(IplImage * image){
	IplImage * background_image = iterative_background_estimation(image,41,1);
	IplImage * binary_image = otsu_algorithm(background_image);
	binary_image = morphological_close(binary_image);
	IplImage *compensated_img = compensate_contrast_variation_modified(image,background_image,binary_image);
	IplImage * otsu_image = otsu_algorithm_modified(compensated_img);
	return otsu_image;

}

