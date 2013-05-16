#include "FunctionsLib.h"
#include <stdexcept>

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

unsigned char * imagelibrary::iplimage_to_bytearray(IplImage * image, int bytes_per_line){
	//unsigned char * rimage= new unsigned char[bytes_per_line*image->height];
	if(bytes_per_line < 0){
		bytes_per_line = image->widthStep/sizeof(uchar);
	}
	unsigned char * rimage = (unsigned char *)malloc(image->width * image->height * sizeof(unsigned char));

	int height = image->height;
	int width = image->width;

	for (int x = 0; x < image->width; x++) {
		for (int y = 0; y < image->height; y++) {
			unsigned char pixel = get_pixel(image,y,x);
			rimage[bytes_per_line * y + x] = 0;//pixel;
		}
	}
	return rimage;

}

namespace cvt
{
	
	bool sorting_function (int i,int j) { return (i<j); }

	void display_image(char * winName,IplImage* image){
		cvNamedWindow(winName,0);
		cvShowImage(winName,image);
	}

	IplImage * iterative_background_estimation(IplImage* image, int kernelRadius,int iterations){
		IplImage * bimg = cvCreateImage(cvSize(image->width, image->height), image->depth, image->nChannels);
		cvSmooth(image, bimg, 3, kernelRadius, kernelRadius);
		for(int i=1;i<iterations;i++){
			cvSmooth(bimg, bimg, 3, kernelRadius, kernelRadius);
			//cvSmooth(bimg, bimg, 3, kernelRadius, kernelRadius);
		}
		return bimg;
	}

	int get_pixel(IplImage * image, int row,int col){ //row,col
		//CvScalar pixel = cvGet2D(image,row,col);
		//int value = pixel.val[0];
		//return value;
		int value = ((uchar*)(image->imageData + image->widthStep*row))[col];
		return value;
	}
	void set_pixel(IplImage * image, int row,int col, int new_value){ 
		((uchar*)(image->imageData + image->widthStep*row))[col]= (uchar)new_value;
	}
	void set_pixel(unsigned char * image_data, int bytes_per_line,int y, int x, unsigned char new_value){
		image_data[bytes_per_line* y + x] = new_value;
	}
	float get_median(vector<int> list){
		if(list.size()==1)
			return (float)list.at(0);

		sort (list.begin(), list.end(), sorting_function);
		int size = list.size();
		int middle = size/2;
		float median;
		if (size%2==0) 
			median = (float)(list.at(middle-1)+list.at(middle))/2;
		else 
			median = (float)(list.at(middle));

		return median;
	}

	int calculate_median(IplImage *image){
		vector<int> values;
		for(int i=0; i<image->height;i++){
			for(int j=0;j<image->width;j++){
				int value = get_pixel(image,i,j);
				values.push_back(value);
			}
		}
		float median = get_median(values);
		int result = ceil(median);
		return result;
	}
	int get_max_value(IplImage *image, IplImage * backimage,double C){
		int max_val = 0;
		for(int i=0; i<image->height;i++){
			for(int j=0;j<image->width; j++){
				double I = get_pixel(image,i,j);
				double BG = get_pixel(backimage,i,j);
				int I_hat = ceil(C/BG * I);
				if(I_hat > max_val)
					max_val = I_hat;
			}
		}
		return max_val;

	}
	int get_max_value(IplImage *image){
		int max_val = 0;
		for(int i=0; i<image->height;i++){
			for(int j=0;j<image->width; j++){
				int pixel =  get_pixel(image,i,j);
				if(pixel > max_val)
					max_val = pixel;
			}
		}
		return max_val;

	}


	IplImage * compensate_contrast_variation(IplImage *image, IplImage * backimage){ //Lu's implementation
		IplImage * rimage = cvCloneImage(image);
		double C = calculate_median(image);

		for(int i=0; i<image->height;i++){
			for(int j=0;j<image->width; j++){
				double I = get_pixel(image,i,j);
				double BG = get_pixel(backimage,i,j);
				int I_hat = ceil(C/BG * I);

				if(I_hat > 255){
					I_hat =255;
				}
				//cvSet2D(rimage,i,j,cvScalar(I_hat,I_hat,I_hat));
				set_pixel(rimage,i,j,I_hat);
			}
		}
		return rimage;

	}
	IplImage * compensate_contrast_interactive(IplImage *image, IplImage * backimage, int C){ 
		IplImage * rimage = cvCloneImage(image);

		for(int i=0; i<image->height;i++){
			for(int j=0;j<image->width; j++){
				double I = get_pixel(image,i,j);
				double BG = get_pixel(backimage,i,j);
				int I_hat = ceil(((double)C/BG) * I);

				if(I_hat > 255){
					I_hat =255;
				}
				set_pixel(rimage,i,j,I_hat);
			}
		}
		return rimage;

	}
	IplImage * compensate_contrast_variation_normalized(IplImage *image, IplImage * backimage){
		IplImage * rimage = cvCloneImage(image);
		double C = 128;//calculate_median(image);
		int max_val = get_max_value(image,backimage,C);

		for(int i=0; i<image->height;i++){
			for(int j=0;j<image->width; j++){
				double I = get_pixel(image,i,j);
				double BG = get_pixel(backimage,i,j);
				int I_hat = ceil(C/BG * I);
				I_hat = ((float)255/max_val) * I_hat;
				cvSet2D(rimage,i,j,cvScalar(I_hat,I_hat,I_hat));
				
			}
		}
		return rimage;

	}

	IplImage * create_white_image(int width,int height,int channels){
		IplImage * rimage = cvCreateImage(cvSize(width ,height),IPL_DEPTH_8U,channels);
		for(int i=0; i < height;i++){
			for(int j=0; j < width; j++){
				cvSet2D(rimage,i,j,cvScalar(255,255,255));
			}
		}
		return rimage;
	}
	
	IplImage* background_subtraction( IplImage* image ,IplImage* back_image){
		IplImage * rimage  =cvCreateImage( cvSize(image->width, image->height), image->depth, image->nChannels);  

		for(int i=0;i<image->height;i++){
			for(int j=0;j<image->width;j++){
				int opixel = get_pixel(image,i,j);
				int bpixel = get_pixel(back_image,i,j);
				int rpixel =  bpixel - opixel;
				if(rpixel<0)
					cvSet2D(rimage,i,j,cvScalar(0,0,0));
				else 
					cvSet2D(rimage,i,j,cvScalar(rpixel,rpixel,rpixel));

			}
		}

		return rimage;


	}

	IplImage* invert_image(IplImage * image){
		IplImage * rimage  =cvCreateImage( cvSize(image->width, image->height), image->depth, image->nChannels);

		for(int y = 0; y < image->height; ++y){
			for(int x = 0; x < image->width; ++x){
				int pixel = get_pixel(image,y,x);
				int ivalue = 255 - pixel;
				cvSet2D(rimage,y,x,cvScalar(ivalue,ivalue,ivalue));
			}
		}
		return rimage;
	}
	
	IplImage* bilateral_filter(IplImage * image, int spacestdv,int rangestdv){
		BilateralFilter *bilateral = new BilateralFilter(image,spacestdv,rangestdv,false);
		IplImage * rimage = bilateral->runFilter();
		return rimage;
	}

	int get_patch_difference(IplImage * temp_image,IplImage * compensated_img, IplImage * back_img,int color=0){
        float difference = 0;
        float num_pixels = 0;
		for(int i=0; i<temp_image->height;i++){
			for( int j=0;j < temp_image->width;j++){
				int pixel = get_pixel(temp_image,i,j);
				if(pixel == color){
					float bpixel = get_pixel(back_img,i,j);
					float cpixel = get_pixel(compensated_img,i,j);
                    difference= difference + fabs(bpixel-cpixel);
                    num_pixels++;
				}
			}
		}
		int result = ceil(difference/num_pixels);
		return result;
	}

	int get_patch_difference_ninas(IplImage * temp_image,IplImage * image, IplImage * back_img,int &num_pixels,int color=0){
        float difference = 0;

		for(int i=0; i<temp_image->height;i++){
			for( int j=0;j < temp_image->width;j++){
				int pixel = get_pixel(temp_image,i,j);
				if(pixel == color){
					float bpixel = get_pixel(back_img,i,j);
					float cpixel = get_pixel(image,i,j);
                    difference= difference + fabs(bpixel-cpixel);
                    num_pixels++;
				}
			}
		}
		int result = ceil(difference/num_pixels);
		return result;
	}

	IplImage * despeckle(IplImage* timage,IplImage * image,IplImage * back_img){ //original despeckle algorithm in Lu's paper
		IplImage * tempimage = cvCloneImage(timage);
		IplImage * rimage = cvCloneImage(timage);
		int color = 100;
		int * hist=create_histogram(256);
		for(int i=0;i<timage->height;i++){
			for(int j=0;j<timage->width;j++){
				int pixel = get_pixel(tempimage,i,j);
				if(pixel == 0){
					cvFloodFill(tempimage,cvPoint(j,i),cvScalar(color,255,255),cvScalar(0,0,0),cvScalar(0,0,0));
					int difference = get_patch_difference(tempimage,image, back_img, color);
					hist[difference]++;
					cvFloodFill(tempimage,cvPoint(j,i),cvScalar(255,255,255),cvScalar(0,0,0),cvScalar(0,0,0));
				}
			}
		}
		int threshold = otsu_threshold_histogram(hist);
		rimage = despeckle_threshold(timage,image,back_img,threshold);

		return rimage;
	}
	
	IplImage * despeckle_threshold_ninas(IplImage* timage,IplImage * compensated_img,IplImage * back_img,int threshold_value, int threshold_size){
		IplImage * tempimage = cvCloneImage(timage);
		IplImage * rimage = cvCloneImage(timage);
		int color = 100;
		for(int i=0;i<timage->height;i++){
			for(int j=0;j<timage->width;j++){
				int pixel = get_pixel(tempimage,i,j);
				if(pixel == 0){
					cvFloodFill(tempimage,cvPoint(j,i),cvScalar(color,255,255),cvScalar(0,0,0),cvScalar(0,0,0));
					int num_pixels = 0;
					int difference = get_patch_difference_ninas(tempimage,compensated_img, back_img,num_pixels, color);
					if(difference < threshold_value && num_pixels < threshold_size)
						cvFloodFill(rimage,cvPoint(j,i),cvScalar(255,255,255),cvScalar(0,0,0),cvScalar(0,0,0));
					cvFloodFill(tempimage,cvPoint(j,i),cvScalar(255,255,255),cvScalar(0,0,0),cvScalar(0,0,0));
				}
			}
		}
		return rimage;
	}

	int get_max_num_pixels(IplImage* timage,IplImage * image,IplImage * back_img){
		IplImage * tempimage = cvCloneImage(timage);
		int color = 100;
		int max_num_pixels = 0;
		for(int i=0;i<timage->height;i++){
			for(int j=0;j<timage->width;j++){
				int pixel = get_pixel(tempimage,i,j);
				if(pixel == 0){
					cvFloodFill(tempimage,cvPoint(j,i),cvScalar(color,255,255),cvScalar(0,0,0),cvScalar(0,0,0));
					int num_pixels = 0;
					get_patch_difference_ninas(tempimage,image, back_img,num_pixels, color);
					if(num_pixels > max_num_pixels) 
						max_num_pixels = num_pixels;
					
					cvFloodFill(tempimage,cvPoint(j,i),cvScalar(255,255,255),cvScalar(0,0,0),cvScalar(0,0,0));
				}
			}
		}
		return max_num_pixels;
	}

	IplImage * despeckle_ninas(IplImage* timage,IplImage * image,IplImage * back_img){ // used in the ninas algorithm submitted to icdar
		IplImage * tempimage = cvCloneImage(timage);
		IplImage * rimage = cvCloneImage(timage);
		int color = 100;
		int * hist=create_histogram(256);
		int * hist_size=create_histogram(256);
		int total_num_pixels = 0;
		int num_cc =0;
		int max_num_pixels = get_max_num_pixels(timage,image,back_img);
		for(int i=0;i<timage->height;i++){
			for(int j=0;j<timage->width;j++){
				int pixel = get_pixel(tempimage,i,j);
				if(pixel == 0){
					cvFloodFill(tempimage,cvPoint(j,i),cvScalar(color,255,255),cvScalar(0,0,0),cvScalar(0,0,0));
					int num_pixels = 0;
					int difference = get_patch_difference_ninas(tempimage,image, back_img,num_pixels, color);
					int value_size = ((float)255/max_num_pixels)* num_pixels;
					hist[difference]++;
					hist_size[value_size]++;
					
					cvFloodFill(tempimage,cvPoint(j,i),cvScalar(255,255,255),cvScalar(0,0,0),cvScalar(0,0,0));
				}
			}
		}
		int threshold_size = otsu_threshold_histogram(hist_size);
		int threshold_value = otsu_threshold_histogram(hist);
		rimage = despeckle_threshold_ninas(timage,image,back_img,threshold_value,threshold_size);

		return rimage;
	}

	IplImage * despeckle_ninas_old(IplImage* timage,IplImage * image,IplImage * back_img){ //experimental algorithm the threshold size gets calculated by a heuristics of dividing the total num of pixels over the numm of ccs
		IplImage * tempimage = cvCloneImage(timage);
		IplImage * rimage = cvCloneImage(timage);
		int color = 100;
		int * hist=create_histogram(256);
		int * hist_size=create_histogram(256);
		float total_num_pixels = 0;
		int num_cc =0;
		int max_num_pixels = get_max_num_pixels(timage,image,back_img);
		for(int i=0;i<timage->height;i++){
			for(int j=0;j<timage->width;j++){
				int pixel = get_pixel(tempimage,i,j);
				if(pixel == 0){
					cvFloodFill(tempimage,cvPoint(j,i),cvScalar(color,255,255),cvScalar(0,0,0),cvScalar(0,0,0));
					int num_pixels = 0;
					int difference = get_patch_difference_ninas(tempimage,image, back_img,num_pixels, color);
					if(num_pixels > 0)
						num_cc++;
					
					total_num_pixels = total_num_pixels + num_pixels;
					hist[difference]++;
					cvFloodFill(tempimage,cvPoint(j,i),cvScalar(255,255,255),cvScalar(0,0,0),cvScalar(0,0,0));
				}
			}
		}

		int threshold_size = total_num_pixels/num_cc;
		int threshold_value = otsu_threshold_histogram(hist);
		rimage = despeckle_threshold_ninas(timage,image,back_img,threshold_value,threshold_size);

		return rimage;
	}


	IplImage * despeckle_threshold(IplImage* timage,IplImage * compensated_img,IplImage * back_img,int threshold){
		float k = .6; // variable suggested to be used by the author upon emailing him   
		IplImage * tempimage = cvCloneImage(timage);
		IplImage * rimage = cvCloneImage(timage);
		int color = 100;
		for(int i=0;i<timage->height;i++){
			for(int j=0;j<timage->width;j++){
				int pixel = get_pixel(tempimage,i,j);
				if(pixel == 0){
					cvFloodFill(tempimage,cvPoint(j,i),cvScalar(color,255,255),cvScalar(0,0,0),cvScalar(0,0,0));
					int difference = get_patch_difference(tempimage,compensated_img, back_img,color);
					if(difference < (threshold*k))
						cvFloodFill(rimage,cvPoint(j,i),cvScalar(255,255,255),cvScalar(0,0,0),cvScalar(0,0,0));

					cvFloodFill(tempimage,cvPoint(j,i),cvScalar(255,255,255),cvScalar(0,0,0),cvScalar(0,0,0));
				}
			}
		}
		return rimage;
	}


	int * create_histogram(int size){
		//int * histogram = new int[size];
		//clear_histogram(histogram,size);
		int* hist = (int*)malloc(sizeof(int)*size);
		memset(hist, 0, sizeof(int)*size);
		return hist;
	}

	int otsu_threshold_histogram(int *hist){ //This has a bug that assumes the lenght of the histogram is always 256. Otherwise it will fail
		int T;
		double V1, V2, M1, M2, Q1, Q2; // variances, means, and probs of groups 1<=t<2
		float Vw; // within-class variance
		float VwBest;
		int Tbest, Tmin, Tmax;
		int i;

		Tmin = 0;
		while((hist[Tmin] == 0) && (Tmin < 255))
			Tmin++;

		Tmax = 255;
		while((hist[Tmax] == 0) && (Tmax > 0))
			Tmax--;

		Tbest = Tmin;
		VwBest = 0.; /* initialize to suppress warning (this gets set when T==Tmin)*/

		for(T = Tmin; T <= Tmax; ++T){
			Q1 = Q2 = M1 = M2 = V1 = V2 = 0.;
			for(i = 0; i <= T; i++)
				Q1 += hist[i];
			for(i = T+1; i < 256; i++)
				Q2 += hist[i];
			for(i = 0; i <= T; i++)
				if(Q1 > 0.)
					M1 += i * hist[i] / Q1;
			for(i = T+1; i < 256; i++)
				if(Q2 > 0.)
					M2 += i * hist[i] / Q2;

			for(i = 0; i <= T; i++)
				if(Q1 > 0.)
					V1 += (i - M1) * (i - M1) * hist[i] / Q1;
			for(i = T+1; i < 256; i++)
				if(Q2 > 0.)
					V2 += (i - M2) * (i - M2) * hist[i] / Q2;

			Vw = Q1 * V1 + Q2 * V2;
			if((Vw <= VwBest) || (T==Tmin)){
				VwBest = Vw;
				Tbest = T;
			}


		}
		return Tbest;
	}



	void clear_histogram(int * histogram,int size){
		//memset(histogram,0,size*sizeof(int));//You could do it this way instead
		for(int i=0;i<size;i++){
			histogram[i] = 0;
		}

	}

	IplImage * apply_threshold(IplImage* image,int threshold, int type){  //type -1 means sign is less than, 1 is greater than, default value =-1
		IplImage* rimage = cvCloneImage(image);

		for(int y = 0; y < image->height; y++){
			for(int x = 0; x < image->width; x++){
				int pixel =get_pixel(image,y,x);
				if(type>0){
					if(pixel>threshold){
						//cvSet2D(rimage,y,x,cvScalar(0,0,0));
						set_pixel(rimage,y,x,0);
					}
					else{
						//cvSet2D(rimage,y,x,cvScalar(255,255,255));
						set_pixel(rimage,y,x,255);
					}
				}
				if(type<0){  
					if(pixel<threshold){// add the equal sign here to increase recall percentile this is a bug that will make dicta_algorithm work
						//cvSet2D(rimage,y,x,cvScalar(0,0,0));
						set_pixel(rimage,y,x,0);
					}
					else{
						set_pixel(rimage,y,x,255);
						//cvSet2D(rimage,y,x,cvScalar(255,255,255));
					}
				}

			}		  
		}
		return rimage;
	}

	IplImage * convert_pixels_to_white(IplImage* image,int threshold){
		IplImage* rimage = cvCloneImage(image);
		for(int y = 0; y < image->height; y++){
			for(int x = 0; x < image->width; x++){
				int pixel =get_pixel(image,y,x);
				if(pixel>threshold)
					cvSet2D(rimage,y,x,cvScalar(255,255,255));
			}		  
		}
		return rimage;
	}

	int * get_histogram(IplImage * image){
		int size = 256;
		int * histogram = create_histogram(size); 
		for(int y = 0; y < image->height; y++){
			for(int x = 0; x < image->width; x++){
				int pixel_value =get_pixel(image,y,x);
				histogram[pixel_value]= histogram[pixel_value] + 1;
			}		  
		}
		return histogram;
	}


	int * get_histogram_no_white(IplImage * image){
		int size = 256;
		int * histogram = create_histogram(size); 
		
		for(int y = 0; y < image->height; y++){
			for(int x = 0; x < image->width; x++){
				int pixel_value =get_pixel(image,y,x);
				if(pixel_value!=255)
					histogram[pixel_value]= histogram[pixel_value] + 1;

			}		  
		}

		return histogram;
	}
	
	int count_black_pixels(IplImage * image){
		int counter = 0;
		for(int y = 0; y < image->height; y++){
			for(int x = 0; x < image->width; x++){
				int pixel_value =get_pixel(image,y,x);
				if(pixel_value==0)
					counter++;

			}		  
		}

		return counter;
	}

	IplImage * add_black_pixels(IplImage* image, IplImage * timage){ //add timage black pixels to image
		IplImage* rimage = cvCloneImage(image);

		for(int y = 0; y < image->height; y++){
			for(int x = 0; x < image->width; x++){
				int tpixel =get_pixel(timage,y,x);
				if(tpixel==0)
					cvSet2D(rimage,y,x,cvScalar(0,0,0));
			}		  
		}
		return rimage;
	}

	void print_array(int * a,int size){
		for(int i=0; i <size; i++) {
			printf(" %d ", a[i]);
		}
		printf("\n");
	}

	void print_2d_bool_array(bool ** a,int height,int width){
		for(int i=0; i <height; i++) {
			for(int j=0; j<width; j++){
				printf(" %d ", a[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	
	void clear_2d_bool_array(bool ** a,int height, int width){
		for(int i=0; i <height; i++) {
			for(int j=0; j<width; j++){
				a[i][j]=false;
			}
		}
	}


	
	IplImage * kittler_algorithm(IplImage * image){
		IplImage* thresholded_image = cvCreateImage( cvSize(image->width, image->height), image->depth, image->nChannels); 
		int * hist = get_histogram(image);
		int T = kittler_threshold_histogram(hist);
		thresholded_image = apply_threshold(image,T);
		return thresholded_image;
	}


	int kittler_threshold_histogram(int *hist){
		int T, i;
		float V1, V2, M1, M2, P1, P2,std1,std2; // variances, means, and probs of groups 1<=t<2
		int Tbest, Tmin, Tmax;
		float bmeMin,bme; //Bayes Minimum error

		Tmin = 0;
		while((hist[Tmin] == 0) && (Tmin < 255))
			Tmin++;

		Tmax = 255;
		while((hist[Tmax] == 0) && (Tmax > 0))
			Tmax--;

		Tbest = Tmin;
		bmeMin = 0.; /* initialize to suppress warning (this gets set when T==Tmin)*/
		bme = 0.;//Bayes Minimum Error


		for(T = Tmin; T <= Tmax; ++T){
			P1 = P2 = M1 = M2 = V1 = V2 = std1 = std2 = 0.;
			for(i = 0; i <= T; i++)	P1 += hist[i];
			for(i = T+1; i < 256; i++) P2 += hist[i];
			for(i = 0; i <= T; i++)	if(P1 > 0.)M1 += i * hist[i] / P1;
			for(i = T+1; i < 256; i++) if(P2 > 0.)M2 += i * hist[i] / P2;
			for(i = 0; i <= T; i++) if(P1 > 0.)V1 += (i - M1) * (i - M1) * hist[i] / P1;
			for(i = T+1; i < 256; i++) if(P2 > 0.)V2 += (i - M2) * (i - M2) * hist[i] / P2;
			std1 = sqrt(V1);
			std2 = sqrt(V2);
			if(V1>0&&V2>0){//otherwise it gives us inf error or other errors
				//bme = (P1 * log(std1))+ P2*log(std2) - (P1*log(P1))-(P2*log(P2)); //another way to do this
				bme = 1+2*(P1 * log(std1)+ P2*log(std2)) - (2*(P1*log(P1)+P2*log(P2)));
				//printf("\nT: %d bme: %f P1:%f P2:%f ",T,bme, P1,P2); 
				//printf("\nT: %d bme: %f P1:%f P2:%f std1:%f std2:%f V1:%f V2:%f ",T,bme, P1,P2,std1,std2,V1,V2); 
				if(bme <= bmeMin || (T==Tmin)){
					bmeMin = bme;
					Tbest = T;
				}
			}
		}
		return Tbest;
	}


	IplImage * niblack_algorithm(IplImage * image , int radius ,double K ){

		IplImage *dest  =cvCreateImage( cvSize(image->width, image->height), image->depth, image->nChannels); 
		unsigned char *imgSrc= reinterpret_cast<unsigned char *>(image->imageData);
		int h = image->height;
		int w = image->width ;
		int step= image->widthStep/sizeof(uchar); 
		unsigned char *imgDst= reinterpret_cast<unsigned char *>(dest->imageData);

		double mean; //mean
		double stddev; // standard deviation
		double numKernPxls; // number of pixels in the kernel (window)
		double thresh; // threshold value

		numKernPxls = (radius*2+1); // length of one side of the square kernel
		numKernPxls *= numKernPxls; // to square it

		for(int y = radius; y < (h-radius-1); ++y){
			for(int x = radius; x < (w-radius-1); ++x){
				// calculate the mean
				mean = 0.;
				for(int ky = y-radius; ky <= y+radius; ++ky){
					for(int kx = x-radius; kx <= x+radius; ++kx){
						if(imgSrc[ky*step+kx]==0x100)printf("256");
						mean += (double)imgSrc[ky*step+kx];
					}
				}
				mean /= numKernPxls;

				// calculate the standard deviation
				stddev = 0.;
				for(int ky = y-radius; ky <= y+radius; ++ky){
					for(int kx = x-radius; kx <= x+radius; ++kx){
						stddev += (((double)imgSrc[ky*step+kx] - mean) *((double)imgSrc[ky*step+kx] - mean));
					}
				}
				stddev /= numKernPxls; // this gives us variance
				stddev = sqrt(stddev); // standard deviation is square root of variance

				//now we get the threshold value for this pixel
				thresh = mean + K*stddev;

				// now set the pixel to black if less than thresh, white otherwise
				imgDst[y*step+x] = (imgSrc[y*step+x] < thresh) ? 0x00 : 0xff;
			}// end for x
		}//end for y
		return dest;
	}
	IplImage *  sauvola_algorithm(IplImage * image, int radius, double K, double R ){
		IplImage *dest  =cvCreateImage( cvSize(image->width, image->height), image->depth, image->nChannels); 
		unsigned char *imgSrc= reinterpret_cast<unsigned char *>(image->imageData);
		int h = image->height;
		int w = image->width ;
		int step= image->widthStep/sizeof(uchar); 
		unsigned char *imgDst= reinterpret_cast<unsigned char *>(dest->imageData);
		//const double K = 0.5; // constant used by Niblack
		double mean; //mean
		double stddev; // standard deviation
		double numKernPxls; // number of pixels in the kernel (window)
		double thresh; // threshold value
		//const double R = 128;
		//radius = 21;

		numKernPxls = (radius*2+1); // length of one side of the square kernel
		numKernPxls *= numKernPxls; // to square it

		for(int y = radius; y < (h-radius-1); ++y){
			for(int x = radius; x < (w-radius-1); ++x){
				// calculate the mean
				mean = 0.;
				for(int ky = y-radius; ky <= y+radius; ++ky){
					for(int kx = x-radius; kx <= x+radius; ++kx){
						if(imgSrc[ky*step+kx]==0x100)printf("256");
						mean += (double)imgSrc[ky*step+kx];
					}
				}
				mean /= numKernPxls;
				// calculate the standard deviation
				stddev = 0.;
				for(int ky = y-radius; ky <= y+radius; ++ky){
					for(int kx = x-radius; kx <= x+radius; ++kx){
						stddev += (((double)imgSrc[ky*step+kx] - mean) *((double)imgSrc[ky*step+kx] - mean));
					}
				}
				stddev /= numKernPxls; // this gives us variance
				stddev = sqrt(stddev); // standard deviation is square root of variance

				//now we get the threshold value for this pixel
				//thresh = mean + K*stddev;
				thresh = mean*(1 + K*(stddev/R -1)) ;
				//T=m*[l+k*(s/R-I)]
				// now set the pixel to black if less than thresh, white otherwise
				imgDst[y*step+x] = (imgSrc[y*step+x] < thresh) ? 0x00 : 0xff;
			}// end for x
		}//end for y
		return dest;
	}

	void batch_thresholding(char * directory,char * filenames[],char * tdirectory,char * tfilenames[], int num_files, IplImage * (*function)(IplImage*)){

		for(int i=0; i< num_files;i++){
			char filename[200];
			strcpy(filename,directory);
			strcat(filename, filenames[i] );
			IplImage * image = cvLoadImage(filename,0);
			IplImage * hdibco_img = (*function)(image);
			strcpy(filename,tdirectory);
			strcat(filename, tfilenames[i] ); 
			cvSaveImage(filename,hdibco_img);
			printf("%s Saved\n",filename);

		}

	}


	IplImage* nina_algorithm(IplImage * image){
		IplImage *img_background = iterative_background_estimation(image);
		IplImage * compensated_img = compensate_contrast_variation_normalized(image,img_background);
		int range_parameter = 2;
		int space_parameter =10;
		IplImage * bilateralImg = bilateral_filter(compensated_img, space_parameter, range_parameter);
		int threshold = -1;
		IplImage * finalImage = rotsu_version3(bilateralImg,threshold);
		finalImage = despeckle_ninas(finalImage,image,img_background);
		return finalImage;
	}
	IplImage * bytearray_to_iplimage(unsigned char * image, int width, int height,int bytes_per_line){
		IplImage  * rimage = cvCreateImage(cvSize(width, height),IPL_DEPTH_8U,1);
		//memcpy(rimage->imageData,image,bytes_per_line * height);  //This line is kind of buggy

		for(int x =0; x< width;x++){
			for(int y=0;y<height;y++){
				unsigned char pixel = image[y* bytes_per_line + x];
				set_pixel(rimage,y,x,pixel);
			}
		}
		return rimage;
		//rimage->imageData = (char*)image;
		//return rimage;
	}
	//unsigned char * iplimage_to_bytearray(IplImage * image){
	//	unsigned char * rimage = reinterpret_cast<unsigned char *>(image->imageData);
	//	int height = image->height;
	//	int width = image->width;
	//	int bytes_per_line = image->widthStep/sizeof(uchar);
	//	return rimage;
	//}
	unsigned char * iplimage_to_bytearray(IplImage * image, int bytes_per_line){
		//unsigned char * rimage= new unsigned char[bytes_per_line*image->height];
		if(bytes_per_line < 0){
			bytes_per_line = image->widthStep/sizeof(uchar);
		}
		unsigned char * rimage = (unsigned char * )calloc(bytes_per_line * image->height,sizeof(unsigned char));
		//unsigned char * rimage = (unsigned char * )calloc(image->width * image->height, sizeof(unsigned char));
		int height = image->height;
		int width = image->width;

		for (int x = 0; x < image->width; x++) {
			for (int y = 0; y < image->height; y++) {
				unsigned char pixel = get_pixel(image,y,x);
				rimage[bytes_per_line * y + x] = pixel;
			}
		}
		
		
		return rimage;

	}

	int * get_fast_histogram(unsigned char* image_data, int width, int height, int bytes_per_line){
		int* histogram = new int[256];
		memset(histogram, 0, sizeof(int)*256);
		for (int y = 0; y < height; y++) {
			unsigned char* row = &image_data[y*bytes_per_line];
			for (int x = 0; x < width; x++) {
				histogram[row[x]]++;
			}
		}
		return histogram;
	}

	unsigned char * apply_threshold(unsigned char * image_data, int width, int height, int bytes_per_line, int threshold){
		unsigned char* thresholded_img = (unsigned char *)malloc(bytes_per_line * height );
		
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				unsigned char intensity = get_pixel(image_data,bytes_per_line,y,x);
				thresholded_img[bytes_per_line * y + x] = intensity < threshold ? 0x00 : 0xff;
			}
		}
		
		return thresholded_img;
	}

	unsigned char get_pixel(unsigned char * image_data,int bytes_per_line, int y, int x){
		unsigned char value = image_data[y*bytes_per_line+x];
		return value ;
	}

	IplImage * otsu_algorithm(IplImage * image){
		int * hist = get_histogram(image);
		int T = otsu_threshold_histogram(hist);
		IplImage * thresholded_img = apply_threshold(image,T);
		return thresholded_img;
	}

	unsigned char * otsu_algorithm(unsigned char * image_data,int width ,int height, int bytes_per_line){
		int * hist = get_fast_histogram(image_data,width,height,bytes_per_line);
		int T = otsu_threshold_histogram(hist);
		unsigned char * thresholded_img = apply_threshold(image_data,width,height,bytes_per_line,T);
		free(hist);
		return thresholded_img;
	}




	void process_lsi_columns(unsigned char* src, int width, int height, int bytesPerLine, int threshold, unsigned char * lsiMap){


		for (int x = 0; x < width; x++) {
			unsigned char runLength = 0;
			for (int y = 0; y < height; y++) {
				unsigned char intensity = src[bytesPerLine * y + x];
				if (intensity <= threshold) {
					lsiMap[width * y + x] = runLength == 255 ? runLength : ++runLength;
				} else {
					runLength = 0;
					lsiMap[width * y + x] = 0;
				}
			}

			// Back propagate the maximum run-length value to all adjacent
			// pixels.
			runLength = 0;
			for (int y = height - 1; y >= 0; y--) {
				unsigned char curLength = lsiMap[width * y + x];
				if (curLength > 0) {
					if (runLength > 0) {
						lsiMap[width * y + x] = runLength;
					} else {
						runLength = curLength;
					}
				} else {
					runLength = 0;
				}
			}
		}
	}

	void process_lsi_rows(unsigned char* src, int width, int height, int bytesPerLine, int threshold, unsigned char * lsiMap){
		// The LSI value for each pixel is determined to be the minimum
		// horizontal and vertical line thickness. Because lsiImage has already
		// been filled with vertical line thickness, store the row results into
		// a temporary buffer until the final value(s) for each row can be
		// computed and then combine that with the vertical values in lsiImage.
		unsigned char* curRow = new unsigned char[width];
		for (int y = 0; y < height; y++) {
			unsigned char * image_row = &src[bytesPerLine * y ];
			unsigned char * lsi_row = &lsiMap[bytesPerLine * y ];
			unsigned char runLength = 0;
			for (int x = 0; x < width; x++) {
				int intensity =  image_row[x];
				if (intensity <= threshold) {
					curRow[x] = runLength == 255 ? runLength : ++runLength;
				} else {
					runLength = 0;
					curRow[x] = 0;
				}
			}

			// Back propagate the maximum run-length value to all adjacent pixels.
			runLength = 0;
			for (int x = width - 1; x >= 0; x--) {
				unsigned char curLength = curRow[x];
				if (curLength > 0) {
					if (runLength == 0) {
						runLength = curLength; // Set runlength to the max val
					}
					lsi_row[x] = std::min(runLength,lsi_row[x]);
				} else {
					runLength = 0;
					lsi_row[x] = 0;
				}
			}
		}
		delete [] curRow;
	}
	unsigned char* compute_lsi_map(unsigned char* src, int width, int height, int bytesPerLine, int threshold) {
		unsigned char* lsiMap = (unsigned char *)malloc(width * height * sizeof(unsigned char));
		process_lsi_columns(src,width,height,bytesPerLine,threshold,lsiMap);
		process_lsi_rows(src,width,height,bytesPerLine,threshold,lsiMap);
		return lsiMap;
	}
	int fast_otsu_algorithm(int* histogram) {
		const int histSz = 256;
		int thresh = 0;
		unsigned int total = 0;
		double weighted_total = 0.0;
		double var_between = 0.0;
		double max_var = 0.0;
		double q1, u1, u2;
		double u1_wt, u1_total;
		double u2_wt, u2_total;

		// Calculate the total mean.
		for(int i = 0; i < histSz; i++) {
			weighted_total += (i+1)*histogram[i];
			total += histogram[i];
		}
		//printf("Weighted Total: %d Total: %d\n", weighted_total, total);

		// q1 - the probability for the group with values less than or equal to t.
		// u1 - the mean for the first group
		// u2 - the mean for the second group
		// The goal is to find the threshold (t) which maximizes the between-group
		// variance (var_between).
		q1 = 0;
		u1_wt = 0;
		u1_total = 0;
		u2_wt = weighted_total;
		u2_total = total;
		for(unsigned int t = 0; t < histSz-1; t++) {
			// q1(t+1) = q1(t) + P(t+1)
			// P(I) represents the histogram probabilities of the observed grey
			// values 1,...,I.  The histogram probability is calculated by taking
			// the histogram count h[] and dividing it by the total count in the
			// histogram.
			q1 += (double)histogram[t]/(double)total;
			u1_wt += (t+1)*histogram[t];
			u1_total += histogram[t];
			u2_wt -= (t+1)*histogram[t];
			u2_total -= histogram[t];
			u1 = u1_wt/u1_total;
			if(u2_total == 0)
				u2 = 0;
			else
				u2 = u2_wt/u2_total;
			var_between = q1*(1-q1)*(u1-u2)*(u1-u2);
			//printf("t: %d q1: %.4f u1: %.4f u2: %.4f ob: %.4f\n", t, q1, u1, u2, var_between);
			if(var_between > max_var) {
				max_var = var_between;
				thresh = t;
			}
		}

		return thresh;
	}

	unsigned char * get_otsu_lsi_map(unsigned char* image_data, int width, int height,int bytes_per_line){
		int * hist = get_fast_histogram(image_data,width,height,bytes_per_line);
		int threshold = fast_otsu_algorithm(hist);
		unsigned char * lsi_map = compute_lsi_map(image_data,width,height,bytes_per_line,threshold);
		return lsi_map;
	}


	void process_lsi_columns(IplImage * image,unsigned char * lsiMap,int bytes_per_line){
		for (int x = 0; x < image->width; x++) {
			unsigned char runLength = 0;
			for (int y = 0; y < image->height; y++) {
				unsigned char pixel = get_pixel(image,y,x);
				if (pixel == 0) {
					lsiMap[bytes_per_line * y + x] = runLength == 255 ? runLength : ++runLength;
				} else {
					runLength = 0;
					lsiMap[bytes_per_line * y + x] = 0;
				}
			}

			// Back propagate the maximum run-length value to all adjacent
			// pixels.
			runLength = 0;
			for (int y = image->height - 1; y >= 0; y--) {
				unsigned char curLength = lsiMap[bytes_per_line * y + x];
				if (curLength > 0) {
					if (runLength > 0) {
						lsiMap[bytes_per_line * y + x] = runLength;
					} else {
						runLength = curLength;
					}
				} else {
					runLength = 0;
				}
			}
		}
	}

	void process_lsi_rows(IplImage * image , unsigned char * lsiMap,int bytes_per_line){
		// The LSI value for each pixel is determined to be the minimum
		// horizontal and vertical line thickness. Because lsiImage has already
		// been filled with vertical line thickness, store the row results into
		// a temporary buffer until the final value(s) for each row can be
		// computed and then combine that with the vertical values in lsiImage.
		unsigned char* curRow = new unsigned char[bytes_per_line];
		for (int y = 0; y < image->height; y++) {
			unsigned char runLength = 0;
			for (int x = 0; x < image->width; x++) {
				int pixel =  get_pixel(image,y,x);
				if (pixel == 0) {
					curRow[x] = runLength == 255 ? runLength : ++runLength;
				} else {
					runLength = 0;
					curRow[x] = 0;
				}
			}

			// Back propagate the maximum run-length value to all adjacent pixels.
			runLength = 0;
			for (int x = image->width - 1; x >= 0; x--) {
				unsigned char curLength = curRow[x];
				if (curLength > 0) {
					if (runLength == 0) {
						runLength = curLength; // Set runlength to the max val
					}
					//lsi_row[x] = std::min(runLength,lsi_row[x]);
					lsiMap[bytes_per_line * y + x ] = std::min(runLength,lsiMap[bytes_per_line * y + x ]);
				} else {
					runLength = 0;
					lsiMap[bytes_per_line * y + x ] = 0;
				}
			}
		}
		delete [] curRow;
	}



	unsigned char* get_lsi_map(IplImage * binary_image,int bytes_per_line) {
		unsigned char* lsiMap = (unsigned char *)malloc(binary_image->width * bytes_per_line);
		//unsigned char* lsiMap = (unsigned char *)malloc(binary_image->width * binary_image->height * sizeof(unsigned char));
		process_lsi_columns(binary_image,lsiMap,bytes_per_line);
		process_lsi_rows(binary_image,lsiMap,bytes_per_line);
		return lsiMap;
	}
	
	IplImage * paint_lsi(unsigned char *  image_data,int width, int height,int bytes_per_line){
		IplImage * rimage = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,3);
		for (int x =0;x<width;x++){
			for(int y=0;y<height;y++){
				unsigned char intensity = get_pixel(image_data,bytes_per_line,y,x);
				if(intensity>0)
				{
					if(intensity<2)
						cvSet2D(rimage,y,x,cvScalar(0,0,255));
					else
						cvSet2D(rimage,y,x,cvScalar(0,255,0));
				}
				else{
					cvSet2D(rimage,y,x,cvScalar(0,0,0));
				}
			}
		}
		return rimage;
	}

	IplImage * morphological_close(IplImage * binary_image,int size){
		IplImage * rimage = cvCreateImage(cvGetSize(binary_image),binary_image->depth,binary_image->nChannels);
		IplImage * tempimage = cvCreateImage(cvGetSize(binary_image),binary_image->depth,binary_image->nChannels);
		IplConvKernel*  kernel = cvCreateStructuringElementEx(size,size,1,1,CV_SHAPE_RECT,NULL); //not released yet
		cvMorphologyEx(binary_image,rimage,tempimage,kernel,CV_MOP_DILATE,1);
		cvReleaseImage(&tempimage);
		return rimage;
	}

	int * get_histogram_modified_nocompensation(IplImage * image, IplImage * binary_image){
		int size = 256;
		int * histogram = create_histogram(size); 
		uchar* srcRow, *binaRow;
		for(int y = 0; y < image->height; y++){
			srcRow = get_row_data(image,y);
			binaRow = get_row_data(binary_image,y);
			for(int x = 0; x < image->width; x++){
				//int pixel_value =get_pixel(binary_image,y,x);
				uchar pixel_value = get_pixel_byte(binaRow,x);
				if(pixel_value!=0){
					pixel_value = get_pixel_byte(srcRow,x);
					histogram[pixel_value]= histogram[pixel_value] + 1;
				}
			}		  
		}
		return histogram;
	}
	IplImage * apply_threshold_modified_nocompensation(IplImage* image,IplImage* binary_image,int threshold){  //type -1 means sign is less than 1 is greater than

		IplImage* rimage = cvCreateImage(cvGetSize(image),IPL_DEPTH_8U,1);
		uchar* binaRow, *srcRow, *dstRow;

		for(int y = 0; y < image->height; y++){
			binaRow = get_row_data(binary_image,y);
			srcRow = get_row_data(image,y);
			dstRow = get_row_data(rimage,y);
			for(int x = 0; x < image->width; x++){
				//int pixel =get_pixel(binary_image,y,x);
				int pixel = get_pixel_byte(binaRow,x);

				if(pixel != 0)
				{
					//pixel =get_pixel(image,y,x);
					pixel = get_pixel_byte(srcRow,x);

					if(pixel<threshold){// add the equal sign here to increase recall percentile this is a bug that will make dicta_algorithm work
						//set_pixel(rimage,y,x,0);
						set_pixel_byte(dstRow,x,0);
					}
					else{
						//set_pixel(rimage,y,x,255);
						set_pixel_byte(dstRow,x,255);
					}
				}
				else{
					set_pixel_byte(dstRow,x,255);
					//set_pixel(rimage,y,x,255);
				}
			}


		}
		return rimage;
	}

	IplImage * otsu_algorithm_modified_nocompensation(IplImage * image, IplImage * binary_image){
		int * hist = get_histogram_modified_nocompensation(image,binary_image);
		int T = otsu_threshold_histogram(hist);
		printf("T: %d",T);
		IplImage * thresholded_img = apply_threshold_modified_nocompensation(image, binary_image,T);
		return thresholded_img;
	}
	int calculate_median_modified(IplImage *image, IplImage * binary_image){
		vector<int> values;
		for(int i=0; i<image->height;i++){
			for(int j=0;j<image->width;j++){
				if(get_pixel(binary_image, i,j)==255)
				{
					int value = get_pixel(image,i,j);
					values.push_back(value);
				}
			}
		}
		float median = get_median(values);
		int result = (int)ceil(median);
		return result;
	}
	IplImage * compensate_contrast_variation_modified(IplImage *image, IplImage * backimage, IplImage * border_image){ //Lu's implementation
		IplImage * rimage = cvCloneImage(image);
		double C = calculate_median_modified(image, border_image);

		for(int i=0; i<image->height;i++){
			for(int j=0;j<image->width; j++){
				double I = get_pixel(image,i,j);
				double BG;
				int I_hat;
				if(get_pixel(border_image,i,j)==255){
					BG = get_pixel(backimage,i,j);
					I_hat = (int)ceil(C/BG * I);
				}
				else{
					I_hat= 0;
				}

				if(I_hat > 255){
					I_hat =255;
				}
				set_pixel(rimage,i,j,I_hat);
			}
		}
		return rimage;

	}
	int * get_histogram_modified(IplImage * image){
		int size = 256;
		int * histogram = create_histogram(size); 
		for(int y = 0; y < image->height; y++){
			for(int x = 0; x < image->width; x++){
				int pixel_value =get_pixel(image,y,x);
				if(pixel_value!=0){
					histogram[pixel_value]= histogram[pixel_value] + 1;
				}
			}		  
		}
		return histogram;
	}
	IplImage * apply_threshold_modified(IplImage* image,int threshold){  

		IplImage* rimage = cvCloneImage(image);
		for(int y = 0; y < image->height; y++){
			for(int x = 0; x < image->width; x++){
				int pixel =get_pixel(image,y,x);
				if(pixel != 0)
				{
					if(pixel<threshold){
						set_pixel(rimage,y,x,0);
					}
					else{
						set_pixel(rimage,y,x,255);
					}
				}
				else{
					set_pixel(rimage,y,x,255);
				}
			}


		}
		return rimage;
	}
	IplImage * otsu_algorithm_modified(IplImage * image){
		int * hist = get_histogram_modified(image);
		int T = otsu_threshold_histogram(hist);
		IplImage * thresholded_img = apply_threshold_modified(image,T);
		return thresholded_img;
	}


	//void process_lsi_columns(unsigned char* src, int width, int height, int bytesPerLine, int threshold, unsigned char * lsiMap){
	void process_lsi_columns(unsigned char* src, IplImage * binary_image, int width, int height, int bytesPerLine, int threshold, unsigned char * lsiMap){


		for (int x = 0; x < width; x++) {
			unsigned char runLength = 0;
			for (int y = 0; y < height; y++) {
				//unsigned char intensity = src[bytesPerLine * y + x];
				unsigned char intensity = get_pixel(binary_image,y,x);
				if (intensity <= threshold) {
					lsiMap[width * y + x] = runLength == 255 ? runLength : ++runLength;
				} else {
					runLength = 0;
					lsiMap[width * y + x] = 0;
				}
			}

			// Back propagate the maximum run-length value to all adjacent
			// pixels.
			runLength = 0;
			for (int y = height - 1; y >= 0; y--) {
				unsigned char curLength = lsiMap[width * y + x];
				if (curLength > 0) {
					if (runLength > 0) {
						lsiMap[width * y + x] = runLength;
					} else {
						runLength = curLength;
					}
				} else {
					runLength = 0;
				}
			}
		}
	}

	void process_lsi_rows(unsigned char* src, IplImage * binary_image, int width, int height, int bytesPerLine, int threshold, unsigned char * lsiMap){
		// The LSI value for each pixel is determined to be the minimum
		// horizontal and vertical line thickness. Because lsiImage has already
		// been filled with vertical line thickness, store the row results into
		// a temporary buffer until the final value(s) for each row can be
		// computed and then combine that with the vertical values in lsiImage.
		unsigned char* curRow = new unsigned char[width];
		for (int y = 0; y < height; y++) {
			unsigned char * image_row = &src[bytesPerLine * y ];
			unsigned char * lsi_row = &lsiMap[bytesPerLine * y ];
			unsigned char runLength = 0;
			for (int x = 0; x < width; x++) {
				int intensity =  image_row[x];
				unsigned char bintensity = get_pixel(binary_image,y,x);
				if (intensity <= threshold && bintensity != 0) {
					curRow[x] = runLength == 255 ? runLength : ++runLength;
				} else {
					runLength = 0;
					curRow[x] = 0;
				}
			}

			// Back propagate the maximum run-length value to all adjacent pixels.
			runLength = 0;
			for (int x = width - 1; x >= 0; x--) {
				unsigned char curLength = curRow[x];
				if (curLength > 0) {
					if (runLength == 0) {
						runLength = curLength; // Set runlength to the max val
					}
					lsi_row[x] = std::min(runLength,lsi_row[x]);
				} else {
					runLength = 0;
					lsi_row[x] = 0;
				}
			}
		}
		delete [] curRow;
	}
	unsigned char* compute_lsi_map(unsigned char* src,IplImage * binary_image, int width, int height, int bytesPerLine, int threshold) {
		unsigned char* lsiMap = (unsigned char *)malloc( bytesPerLine * height );
		//unsigned char* lsiMap = (unsigned char *)malloc(width * height * sizeof(unsigned char));
		//process_lsi_columns(src,binary_image,width,height,bytesPerLine,threshold,lsiMap);
		process_lsi_rows(src,binary_image,width,height,bytesPerLine,threshold,lsiMap);
		return lsiMap;
	}

	int * get_fast_histogram_modified(unsigned char* image_data, IplImage * binary_image,int width, int height, int bytes_per_line){
		int* histogram = new int[256];
		memset(histogram, 0, sizeof(int)*256);
		for (int y = 0; y < height; y++) {
			unsigned char * orig_row = &image_data[y*bytes_per_line];
			char * binary_row = &binary_image->imageData[y*bytes_per_line];
			for (int x = 0; x < width; x++) {
				if(binary_row[x]!=0)
					histogram[orig_row[x]]++;
			}
		}
		return histogram;
	}
	unsigned char * otsu_lsi_algorithm_modified_nocompensation(unsigned char * image_data,IplImage * binary_image,int width ,int height, int bytes_per_line){
		int * hist = get_fast_histogram_modified(image_data,binary_image,width,height,bytes_per_line);
		int T = otsu_threshold_histogram(hist);
		unsigned char * thresholded_img = compute_lsi_map(image_data,binary_image,width,height,bytes_per_line,T);
		free(hist);
		return thresholded_img;
	}

	unsigned char *  lsi_algorithm_version5(unsigned char * image_data, int width,int height,int bytes_per_line){
		unsigned char * rimage_data = otsu_algorithm(image_data,width,height,bytes_per_line);
		IplImage * binary_image = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
		binary_image->imageData = (char *)rimage_data;
		binary_image = morphological_close(binary_image,15);
		unsigned char * otsu_image = otsu_lsi_algorithm_modified_nocompensation(image_data,binary_image,width, height, bytes_per_line);
		
		return otsu_image;
		//free(rimage_data);
		//cvReleaseImage(&binary_image);
		//return otsu_image;
	}
	
	bool is_coordinate_inside_boundaries(int y, int x,IplImage * image){
		if(y>-1 && x>-1 && y< image->height && x < image->width)
			return true;
		else 
			return false;
	}

	int get_max_value(vector<int>values){
		int max_val = 0;
		for(int i=0; i<values.size();i++){
			int value = values.at(i);
			if(value > max_val){
				max_val = value;
			}
		}
		return max_val;
	}
	int get_min_value(vector<int>values){ //this function assumes that max val could only be 255
		int min_val = 255;
		for(int i=0; i<values.size();i++){
			int value = values.at(i);
			if(value < min_val){
				min_val = value;
			}
		}
		return min_val;
	}


	
	uchar* get_row_data(IplImage * image,int row){ // sinonym of the getRowData but with a different format coding
		uchar* row_data = ((uchar*)(image->imageData + image->widthStep*row));
		return row_data;
	}
	
}







