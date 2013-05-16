#include "TestingTools.h"
#include <ctime>

using namespace std;
using namespace cvt;

Tester::Tester(){
	this->first_run = true;
}

Tester::Tester(char * file_to_open, char * file_to_save){
	this->file_to_save = file_to_save;
	image = cvLoadImage(file_to_open,0);
	display_image("original", image);
	cvSaveImage("output/original.png",image);
	
}

void Tester::test_batch_thresholding(){
	
	char * directory  = "C:\\Users\\ninao\\Documents\\images\\dibco_test_images\\";
	char * filenames[] = {"H01.bmp","H02.bmp","H03.bmp","H04.bmp","H05.bmp"};
	char * tdirectory  = "C:\\Users\\ninao\\Documents\\images\\results\\lu\\";
	char * tfilenames[] = {"H01_T.png","H02_T.png","H03_T.png","H04_T.png","H05_T.png"};
	int num_files = 5;
	batch_thresholding(directory,filenames,tdirectory,tfilenames,num_files,otsu_algorithm);
	
}

void Tester::time_algorithm(IplImage * (*function)(IplImage*)){
	clock_t start;
	double diff;
	start = clock();
	IplImage * timage = (*function)(image);
	diff = ( clock() - start)/(double)CLOCKS_PER_SEC;
	printf("\nTime taken: %f seconds",diff);
	display_image("",timage);
	cvSaveImage(this->file_to_save,timage);


}
void Tester::start_timer(){
	this->start = clock();
	if(this->first_run){
		this->total_time =clock();
		this->first_run=false;
	}
}
void Tester::stop_timer(){
	double diff = (clock() - this->start)/(double)CLOCKS_PER_SEC;
	printf("\nTime taken: %f seconds\n",diff);
}
void Tester::get_total_testing_time(){
	double diff = (clock()- this->total_time)/(double)CLOCKS_PER_SEC;
	printf("\n\nTotal time taken: %f seconds",diff);
}



