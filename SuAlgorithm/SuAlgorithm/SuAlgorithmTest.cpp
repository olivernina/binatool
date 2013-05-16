#include "stdafx.h"
#include "SuAlgorithmTest.h"


IplImage * image;
Tester * tester = new Tester();

float test_min_max_enhancement(int y,int x,int kernel_radius,IplImage * image,IplImage *min_image, IplImage * max_image){

	//float median = get_median(kernel_values);
	int max_val = get_pixel(max_image,y,x);
	int min_val = get_pixel(min_image,y,x);
	
	int e = .001;
	float new_value = max_val - min_val;
	//float new_value = (float)(max_val - min_val)/(max_val);
	//float new_value = (float)(max_val - min_val)/(median + e);
	//float new_value = (float)(max_val - min_val)/(max_val + e);//seems better
	//float new_value = (float)(max_val - min_val)/(min_val + e);
	//float new_value = (float)(max_val - min_val)/((max_val + min_val)/2 + e);
	//float new_value = (float)(min_val)/(max_val - min_val);
	//float new_value = (float)(max_val - min_val)/(max_val + min_val + e); //original
	//int rvalue = (int)ceil(new_value);
	//return rvalue ;
	return new_value;
}
float test_min_max_enhancement_old(int y,int x,int kernel_radius,IplImage * image){
	vector<int> kernel_values;
	for(int i=y-kernel_radius;i<=y+kernel_radius;i++){
		for(int j=x-kernel_radius;j<=x+kernel_radius;j++){
			if(is_coordinate_inside_boundaries(i,j,image)){
				int value = get_pixel(image,i,j);
				kernel_values.push_back(value);
			}

		}
	}
	float median = get_median(kernel_values);
	int max_val = get_max_value(kernel_values);
	int min_val = get_min_value(kernel_values);
	
	int e = .001;
	//float new_value = (float)(max_val - min_val)/(max_val + min_val + e);
	//float new_value = (float)(max_val - min_val)/(median + e);
	//float new_value = (float)(max_val - min_val)/(max_val + e);//seems better
	float new_value = (float)(max_val - min_val)/(min_val + e);
	//float new_value = (float)(max_val - min_val)/((max_val + min_val)/2 + e);
	//int rvalue = (int)ceil(new_value);
	//return rvalue ;
	return new_value;
}

void test_get_normalization_parameters(IplImage * image,IplImage* min_img, IplImage * max_img,int kernel_radius, float& rmax_value,float& rmin_value){
	float max_value=0;
	float min_value=255;
	for(int y=0;y<image->height;y++){
		for(int x=0;x<image->width;x++){
			float value = test_min_max_enhancement(y,x,kernel_radius,image,min_img,max_img);
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

void test_get_normalization_parameters_old(IplImage * image,int kernel_radius, float& rmax_value,float& rmin_value){
	float max_value=0;
	float min_value=255;
	for(int y=0;y<image->height;y++){
		for(int x=0;x<image->width;x++){
			float value = test_min_max_enhancement_old(y,x,kernel_radius,image);
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

void test_min_max_edge_detection(){
	tester->start_timer();
	float max_value;
	float min_value;
	IplImage * max_image = cvCreateImage(cvGetSize(image),image->depth,image->nChannels); 
	IplImage * min_image = cvCreateImage(cvGetSize(image),image->depth,image->nChannels); 
	int values [] = {0,0,0,0,0,0,0,0,0};
	IplConvKernel * kernel = cvCreateStructuringElementEx(3,3,1,1,CV_SHAPE_RECT,values);
	cvErode(image,min_image,kernel);
	cvDilate(image,max_image,kernel);
	//display_image("erode",min_image);
	//display_image("dilate",max_image);

	int kernel_radius = 1;
	//test_get_normalization_parameters(image,min_image,max_image,kernel_radius,max_value,min_value);

	IplImage * rimage = cvCloneImage(image);
	for(int y=0;y<image->height;y++){

		for(int x=0;x<image->width;x++){
			//int value = get_pixel(image,y,x);
			//printf("%d \n",value);
			float enhanced_value = test_min_max_enhancement(y,x,kernel_radius,image,min_image,max_image); 
			//int new_value = 255 * enhanced_value/max_value;
			int new_value = enhanced_value;
			set_pixel(rimage,y,x,new_value);
		}
	}

	tester->stop_timer();
	display_image("result",rimage);
	cvSaveImage("images/edges.png",rimage);

}


void test_normalization_min_max_old(){
	float max_value;
	float min_value;
	int kernel_radius = 1;
	//get_normalization_parameters(image,kernel_radius,max_value,min_value);
	printf("max: %f, min: %f",max_value,min_value);
}
void test_normalization_min_max(){
	float max_value;
	float min_value;
	IplImage * max_image = cvCreateImage(cvGetSize(image),image->depth,image->nChannels); 
	IplImage * min_image = cvCreateImage(cvGetSize(image),image->depth,image->nChannels); 
	IplConvKernel * kernel = cvCreateStructuringElementEx(3,3,1,1,CV_SHAPE_RECT);
	cvErode(image,min_image,kernel);
	cvDilate(image,max_image,kernel);
	display_image("dilate",min_image);
	display_image("erode",max_image);

	int kernel_radius = 1;
	test_get_normalization_parameters(image,min_image,max_image,kernel_radius,max_value,min_value);
	printf("max: %f, min: %f",max_value,min_value);
}
void test_otsu_step(){
	//IplImage * edge_image = min_max_edge_detection(image);
	//display_image("edge_image",edge_image);
	IplImage * edge_image = cvLoadImage("images/edges.png",0);
	display_image("edges image",edge_image);
	IplImage * otsu_image = otsu_algorithm(edge_image);
	otsu_image = invert_image(otsu_image);
	display_image("otsu image",otsu_image);
	IplImage * segmented_pixels = get_pixels_on_template(image,otsu_image);
	display_image("segmented pixs",segmented_pixels);
	int threshold = otsu_thresholding_no_white(segmented_pixels);
	IplImage* fimage = apply_threshold(segmented_pixels,threshold);
	display_image("fimage", fimage);
	cvSaveImage("images/otsu_edges.png",otsu_image);

	
}

void test_peak_image_values_from_otsu(){
	IplImage * edge_image = cvLoadImage("images/otsu_edges.png",0);
	display_image("edge_images",edge_image);
	//int peak_value = get_max_value(edge_image);
	//printf("peak_val: %d",peak_value);
	//IplImage* rimage = cvCreateImage(cvGetSize(edge_image),edge_image->depth,edge_image->nChannels);
	//cvZero(rimage);
	IplImage * rimage = create_white_image(image->width,image->height,1);
	int kernel_radius = 1;
	for(int x=0;x<rimage->width;x=x+1){
		for(int y=0;y<rimage->height;y++){
			int center_value = get_pixel(edge_image,y,x);
			bool center_is_peak = false;
			/*for(int i=x-kernel_radius;i<=x+kernel_radius;i++){
				if(is_coordinate_inside_boundaries(y,i,image)){
					int pixel = get_pixel(edge_image,y,i);
					if(pixel>center_value){
						center_is_peak = false;
					}
				}
			}*/
			if(center_value ==0)
			if(is_coordinate_inside_boundaries(y,x-kernel_radius,image)&&is_coordinate_inside_boundaries(y,x+kernel_radius,image))
			{
				int left_pixel = get_pixel(edge_image,y,x-kernel_radius);
				int right_pixel = get_pixel(edge_image,y,x+kernel_radius);
				int threshold = 20;
				if(left_pixel == 255 || right_pixel== 255)
					center_is_peak= true;
			}

			if(center_is_peak){
				//set_pixel(rimage,y,x,center_value);
				set_pixel(rimage,y,x,0);
			//if(pixel==255){
				//set_pixel(rimage,y,x,255);
			//}
			}
		}
	}
	//rimage = invert_image(rimage);
	display_image("peak values ",rimage);
	cvSaveImage("images/peakvalues.png",rimage);

}
void test_stroke_width_approzimation(){
	IplImage* peak_values = cvLoadImage("images/peakvalues.png",0);
	display_image("peakimage",peak_values);
	int text_width = text_width_approximation(peak_values);
	printf("text width: %d",text_width);
}



void test_local_thresholding(){
	int text_width = 3;
	IplImage* edges_image = cvLoadImage("images/otsu_edges.png",0);
	display_image("edges image",edges_image);
	IplImage* original_image = image;
	//IplImage* local_thresholding(IplImage * compensate_img, IplImage * edges_img,int text_width){
    int window_size = text_width * 2;
    if(window_size%2==0) // to make it odd
        window_size++;
    
    int radius = floor((float)window_size /2);
    
    int Nmin = window_size;
    IplImage * rimage = cvCloneImage(image);
	int height_limit  = image->height-radius;
	int width_limit = image->width-radius;

	for(int j=0; j< image->height ;j++){ //change radius to 130 ie to debug windows size
		for(int i=0 ; i < image->width;i++){
			if(j <radius || i < radius || j>= height_limit ||i >= width_limit){ //This is to make whie the borders, we assume there is no text in the borders
					set_pixel(rimage,j,i,255);
					continue;
			}

            int Ne = get_number_edge_pixels(edges_image,i,j,radius);
            float Emean = get_edges_mean(edges_image,image,i,j,radius);
			float Estd = get_edges_std(edges_image,image,Emean,i,j,radius);
			int pixel = get_pixel(image,j,i);
			if( Ne >= Nmin && pixel <= (Emean+Estd/2)){
				set_pixel(rimage,j,i,0);
			}
			else{
				set_pixel(rimage,j,i,255);
			}
		}
	}
    
    //return rimage;

	display_image("local thresholding",rimage);
	cvSaveImage("images/localt.png",rimage);

	
}
void test_su_algorithm_parts(){
	IplImage* edges_image = min_max_edge_detection(image);
	display_image("edges image",edges_image);
	IplImage * contrast_image = otsu_algorithm(edges_image);
	contrast_image = invert_image(contrast_image);
	display_image("otsu image",contrast_image);
	IplImage* peak_pixels_image = get_peak_pixels(contrast_image);
	display_image("peak_pixels_image",peak_pixels_image);
	int text_width = text_width_approximation(peak_pixels_image);
	printf("text width: %d",text_width);
	display_image("image before local thresholding",image);
	IplImage* rimage = perform_local_thresholding(image,contrast_image,text_width);
	display_image("resutl",rimage);

}

void test_su_algorithm(){ 
	tester->start_timer();
	IplImage* rimage = su_algorithm(image);
	tester->stop_timer();
	display_image("result",rimage);
	cvSaveImage("image2-T.png",rimage);

}
void test_performance(){
	
	//image = cvLoadImage("C:\\Users\\ninao\\Documents\\images\\testimages\\bigimage.bmp",0);
	tester->start_timer();
	IplImage* rimage = su_algorithm(image);
	tester->stop_timer();
	display_image("result",rimage);
	cvSaveImage("images/bigimage-su.png",rimage);


}
void test_performance_parts(){
	tester->start_timer();
	printf("edge detection\n");
	IplImage* edges_image = min_max_edge_detection(image);
	tester->stop_timer();
	tester->start_timer();
	printf("otsu\n");
	IplImage * contrast_image = otsu_algorithm(edges_image);
	tester->stop_timer();
	tester->start_timer();
	printf("invert image\n");
	contrast_image = invert_image(contrast_image);
	tester->stop_timer();
	tester->start_timer();
	printf("get peak pixels\n");
	IplImage* peak_pixels_image = get_peak_pixels(contrast_image);
	tester->stop_timer();
	tester->start_timer();
	printf("text width\n");
	int text_width = 3;//text_width_approximation(peak_pixels_image);
	printf("text width: %d \n",text_width);
	tester->stop_timer();
	tester->start_timer();
	printf("local thresholding\n");
	IplImage* rimage = perform_local_thresholding(image,contrast_image,text_width);
	tester->stop_timer();
	tester->get_total_testing_time();
	display_image("rimage",rimage);

}

void test_batch_algorithm(){
	char * directory  = "C:\\Users\\ninao\\Documents\\images\\dibco_test_images\\";
	char * filenames[] = {"H01.bmp","H02.bmp","H03.bmp","H04.bmp","H05.bmp"};
	char * tdirectory  = "C:\\Users\\ninao\\Documents\\images\\results\\su\\";
	char * tfilenames[] = {"H01_T.png","H02_T.png","H03_T.png","H04_T.png","H05_T.png"};
	int num_files = 5;
	batch_thresholding(directory,filenames,tdirectory,tfilenames,num_files,su_algorithm);
	
}
void test_time_opencv_erode(){

	//image = cvLoadImage("C:\\Users\\ninao\\Documents\\images\\dibco_test_images\\H04.bmp",0);
	display_image("image",image);
	IplImage * rimage = cvCloneImage(image);
	int values [] = {0,0,0,0,0,0,0,0,0};
	IplConvKernel * kernel = cvCreateStructuringElementEx(3,3,1,1,CV_SHAPE_RECT,values);
	tester->start_timer();
	cvErode(image,rimage,kernel);
	tester->stop_timer();
	display_image("erode",rimage);
}


IplImage* test_background_subtraction( IplImage* image ,IplImage* back_image){
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
void test_backremoval(){
	float max_value;
	float min_value;
	IplImage * rimage = cvCreateImage(cvGetSize(image),image->depth,image->nChannels); 
	IplImage * min_image = cvCreateImage(cvGetSize(image),image->depth,image->nChannels); 
	int values [] = {0,0,0,0,0,0,0,0,0};
	IplConvKernel * kernel = cvCreateStructuringElementEx(3,3,1,1,CV_SHAPE_RECT,values);
	cvDilate(image,rimage,kernel);
	display_image("dilate",rimage);
	cvErode(rimage,rimage,kernel);
	display_image("erode",rimage);
	
	//cvDilate(max_image,max_image,kernel);
	//cvDilate(max_image,max_image,kernel);
	display_image("original",image);
	display_image("rimage",rimage);

	//rimage = test_background_subtraction(image,max_image);
	//display_image("back removed",rimage);

	//IplImage * otsu_image = otsu_algorithm(rimage);
	//otsu_image = invert_image(otsu_image);
	//display_image("otsu image",otsu_image);

}

void test_stable(){

	//image = cvLoadImage("C:\\Users\\ninao\\Documents\\images\\testimages\\bigimage.bmp",0);
	tester->start_timer();
	test_min_max_edge_detection();
	IplImage * edges_image = cvLoadImage("images/edges.png",0);
	display_image("edges",edges_image);
	IplImage * rimage = cvCreateImage(cvGetSize(image),image->depth,image->nChannels); 
	int values [] = {0,0,0,0,0,0,0,0,0};
	IplConvKernel * kernel = cvCreateStructuringElementEx(3,3,1,1,CV_SHAPE_RECT,values);
	//cvDilate(edges_image,rimage,kernel);
	//display_image("dilate",rimage);
	IplImage * otsu_image = otsu_algorithm(edges_image);
	otsu_image = invert_image(otsu_image);
	tester->stop_timer();
	cvSaveImage("images/fasterver.png",otsu_image);
	display_image("otsu image",otsu_image);

}

void test_opencv_min_max(){

	IplImage * rimage = cvCreateImage(cvGetSize(image),image->depth,image->nChannels); 
	IplImage * timage = cvCreateImage(cvGetSize(image),image->depth,image->nChannels); 
	IplConvKernel * kernel = cvCreateStructuringElementEx(3,3,1,1,CV_SHAPE_RECT);
	//cvErode(image,min_image,kernel);
	//cvDilate(image,max_image,kernel);
	//display_image("dilate",min_image);
	//display_image("erode",max_image);
	tester->start_timer();
	cvMorphologyEx(image,rimage,timage,kernel,CV_MOP_GRADIENT);
	tester->stop_timer();
	display_image("edges",rimage);
	cvSaveImage("edges.png",rimage);

}


void test_Su(){
   IplImage* edge_image = min_max_edge_detection(image);
   display_image("edges",edge_image);

   IplImage * contrast_image = otsu_algorithm(edge_image);
   //contrast_image = invert_image(contrast_image);
   display_image("contrast",contrast_image);
   IplImage* peak_pixels_image = get_peak_pixels(contrast_image);
   display_image("peak",peak_pixels_image);
  
   int text_width = text_width_approximation(peak_pixels_image);
   
   printf("text width:%d ",text_width);
   tester->start_timer();
   IplImage* rimage = perform_local_thresholding(image,contrast_image,text_width);
   tester->stop_timer();
	display_image("final image",rimage);
	cvSaveImage("suimage.png",rimage);
   cvReleaseImage(&edge_image);

}

char * wchar_to_string(_TCHAR* widechar)
{
	int size=0;
	while( (char)widechar[size] != '\0'){
		size++;
	}
	size++;
	char * charpointer = new char[size];
	wcstombs(charpointer, widechar, size );
	return charpointer;
}

void run_as_shell_command(int argc, _TCHAR* argv[]){

	if(argc > 2){
		char * filename = wchar_to_string(argv[1]);
		char * filename_result = wchar_to_string(argv[2]);
		printf(filename);
		printf("\n");
		printf(filename_result);
		printf("\n");
		IplImage *input_image = cvLoadImage(filename,0);
		IplImage * rimage = su_algorithm(input_image);
		cvSaveImage(filename_result,rimage);
		cvReleaseImage(&rimage);
	}
	else{
		printf("enter more arguments");
	}
}

int _tmain(int argc, _TCHAR* argv[])
{

	//run_as_shell_command(argc, argv);

	image = cvLoadImage("C:\\Users\\Oliver\\Projects\\FHTW2013\\binatool\\dibco09-images\\H04.bmp",0);
	IplImage * rimage = su_algorithm(image);
	display_image("su",rimage);
	cvWaitKey();


    return 0;

}

