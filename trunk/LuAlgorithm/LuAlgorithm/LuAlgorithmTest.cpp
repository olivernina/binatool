#include "stdafx.h"
#include "FunctionsLib.h"
#include "LuAlgorithmTest.h"
#include "TestingTools.h"

IplImage * image ; 

LuAlgorithmTester::LuAlgorithmTester(IplImage * test_image){
	image = test_image;
}
void test_get_row(IplImage*image){
	get_row_image(image,0);
}
void test_sample_image(){
	vector<int> signal = get_row_image(image,104);
	vector<Coordinates> sampled_signal =sample_signal(signal, 1);
	//coeffs = polynomial_fitting(sampled_signal,6)
}
void test_polynomial_fitting(){
	vector<int> signal = get_row_image(image,104);
	vector<Coordinates> sampled_signal =sample_signal(signal, 3);
	int degree = 6;
	double* coeffs = polynomial_fitting(sampled_signal,degree);
	print_array(coeffs,7);
	vector<int> y = calculate_fit_polynomial(signal,coeffs,degree);
	print_vector(y);
	printf("");
}
void test_iterative_fitting(){
	vector<int> signal = get_row_image(image,104);
	vector<Coordinates> sampled_signal =sample_signal(signal, 3);
	vector<int> y = iterative_fitting(signal,sampled_signal,1000);
	print_vector(y);

}
void test_matrix_operations(){
	int obs=5;
	int degree = 2;
	double dx[] = { 1,  2,  3,  4,  5};
	double dy[] = {6, 11, 18, 27, 38};

	/*y.setlength(obs);
	x.setlength(obs, degree);
	c.setlength(degree);*/

	CvMat* X = cvCreateMat(obs,degree+1,CV_32FC1);
	CvMat* y = cvCreateMat(obs,1,CV_32FC1);
	CvMat* c = cvCreateMat(1,degree,CV_32FC1);
	CvMat* M= cvCreateMat(obs,obs,CV_32FC1);
	CvMat* X_T= cvCreateMat(degree+1,obs,CV_32FC1);
	for(int i = 0; i < obs; i++)
	{
		//x(i,0) = 1.0;
		cvmSet(X,i,0,1.0);
		for(int j=1; j <= degree; j++) {
			//x(i,j) = pow(dx[i], j);
			cvmSet(X,i,j,pow(dx[i], j));
		}
		//y(i) = dy[i];
		cvmSet(y,i,0,dy[i]);
	}
	cvTranspose(X, X_T);
	cvMatMul( X,X_T, M);


}

void LuAlgorithmTester::test_polyfit(){
	double x[] = { 1,  2,  3,  4,  5};
	double y[] = {6,11,18,27,38};

	int i;
	int obs = 5;
	int degree = 1;
	degree = degree + 1;
	double *coeff = new double [degree];

	polyfit(obs, degree, x, y, coeff);
	for(i=0; i < degree; i++) {
		printf("%lf\n", coeff[i]);
	}
	printf("");
}
void test_background_estimation(){
	int ks = 5;
	IplImage * backimage = background_estimation(image,ks);
	cvSaveImage("backest.pgm",backimage);
	display_image("backimage",backimage);
}
void test_back_compensation(){
	char* filename = "backest.pgm";
	IplImage * backimage = cvLoadImage(filename,0);
	display_image("backimage",backimage);
	IplImage *rimage = compensate_contrast_variation(image,backimage);
	display_image("compensated",rimage);
	cvSaveImage("compensated.pgm",rimage);
}
void test_edge_detection(){
	char* filename = "compensated.pgm";
	IplImage * compensatedImg = cvLoadImage(filename,0);
	display_image("compensated image",compensatedImg);
	IplImage * rimage = edge_detection(compensatedImg);
	display_image("edges", rimage);
	cvSaveImage("edges.pgm",rimage);
}
void test_text_width_approximation(){
	char * filename = "edges.pgm";
	IplImage * edges_image = cvLoadImage(filename,0);
	display_image("edges", edges_image);
	int text_width = text_width_approximation(edges_image);
	printf("text width: %d",text_width);

}
void test_distance_stroke(){
	char * filename = "edges.pgm";
	IplImage * edges_image = cvLoadImage(filename,0);
	display_image("edges", edges_image);
	vector<int>row = get_row_image(edges_image,5);
	int * histogram = new int[35];
	clear_histogram(histogram,35);
	get_distance_next_stroke(row,histogram);
	//print_array(histogram,35);
}
void test_thresholding(){
	char * filename = "edges.pgm";
	IplImage * edges_image = cvLoadImage(filename,0);
	IplImage * compensate_img = cvLoadImage("compensated.pgm",0);
	display_image("edges", edges_image);
	display_image("compensated",compensate_img);
	int text_width =3;
	IplImage * rimage = local_thresholding(compensate_img,edges_image,text_width);
	display_image("dibco",rimage);
	cvSaveImage("dibcoalg.pgm",rimage);
}
void test_dispekle(){
	display_image("original",image);
	IplImage * timage = cvLoadImage("dibcoalg.pgm",0);
	display_image("timage",timage);
	IplImage * compensated_img = cvLoadImage("compensated.pgm",0);
	display_image("cimage",compensated_img);
	IplImage * back_img = cvLoadImage("backest.pgm",0);
	display_image("bimg",back_img);
	IplImage * dispekle_img = despeckle(timage,compensated_img,back_img);
	display_image("dispekle",dispekle_img);        
	cvSaveImage("dispekled.pgm",dispekle_img);
}
void test_logical_operators(){
	IplImage* dispekle_img = cvLoadImage("dispekled.pgm",0);
	display_image("dispekle", dispekle_img);
	IplImage * finalImg = apply_logical_operators(dispekle_img);
	display_image("final",finalImg);

}

void LuAlgorithmTester::test_complete_algorithm(){
	int ks = 1;
	cvSaveImage("images/original.jpg",image);
	printf("starting background estimation...\n");
	IplImage * back_img = background_estimation(image,ks);
	//IplImage * back_img = iterative_background_estimation(image);
	display_image("backimage",back_img);
	cvSaveImage("images/background_estimation.jpg",back_img);
	printf("background estimation done\n");
	printf("starting ccv...\n");
	IplImage *compensated_img = compensate_contrast_variation(image,back_img);
	display_image("compimage", compensated_img);
	cvSaveImage("images/compensated_image.jpg",compensated_img);
	printf("ccv completed\n");
	printf("starting background estimation...\n");
	printf("starting edge detection...\n");
	IplImage * edges_image = edge_detection(compensated_img);
	display_image("edges", edges_image);
	cvSaveImage("images/stroke_edge_detection.jpg",edges_image);
	printf("edge detection done\n");
	printf("starting text width approximation...\n");
	int text_width =text_width_approximation(edges_image);
	printf("text width: %d",text_width);
	printf("text width approximation done\n");
	printf("starting local thresholding...\n");
	IplImage * timage = local_thresholding(compensated_img,edges_image,text_width);
	printf("local thresholding done\n");
	display_image("tresholded", timage);
	cvSaveImage("images/local_thresholding.jpg",timage);
	printf("starting dispekle...\n");
	IplImage * dispekle_img = despeckle(timage,image,back_img);
	printf("dispekle done.\n");
	display_image("dispekle", dispekle_img);
	cvSaveImage("images/dispekle.jpg",dispekle_img);
	printf("starting logical operators...\n");
	IplImage * finalImg = apply_logical_operators(dispekle_img);
	printf("logical operators done.\n");
	display_image("final",finalImg);
	cvSaveImage("images/finalimage.pgm",finalImg);
	cvSaveImage("images/finalimage.jpg",finalImg);
}

void test_save_image(){
	cvSaveImage("images/original.jpg",image);
}

void test_batch_thresholding(){
	char * directory  = "C:\\Documents and Settings\\onina\\My Documents\\workspace\\Experiments\\BenchImages\\Test_Images\\";
	char * filenames[] = {"H01.bmp","H02.bmp","H03.bmp","H04.bmp","H05.bmp"};
	char * tdirectory  = "C:\\Documents and Settings\\onina\\My Documents\\workspace\\Experiments\\mymethod\\mymethod\\output\\lusmethod\\";
	char * tfilenames[] = {"H01_T.png","H02_T.png","H03_T.png","H04_T.png","H05_T.png"};
	int num_files = 5;

	//batch_thresholding(directory,filenames,tdirectory,tfilenames,num_files,lu_algorithm);

}

void test_time_lu_algorithm(){
	Tester * tester= new Tester();
	
	int ks = 10;
	//printf("%d",image->width);
	cvSaveImage("images/original.jpg",image);
	printf("starting background estimation...\n");
	tester->start_timer();
	IplImage * back_img = background_estimation(image,ks);
	//IplImage * back_img = iterative_background_estimation(image);
	tester->stop_timer();
	display_image("backimage",back_img);
	cvSaveImage("images/background_estimation.jpg",back_img);
	printf("background estimation done\n");
	printf("starting ccv...\n");
	tester->start_timer();
	IplImage *compensated_img = compensate_contrast_variation(image,back_img);
	tester->stop_timer();
	display_image("compimage", compensated_img);
	cvSaveImage("images/compensated_image.jpg",compensated_img);
	printf("ccv completed\n");
	printf("starting edge detection...\n");
	tester->start_timer();
	IplImage * edges_image = edge_detection(compensated_img);
	tester->stop_timer();
	display_image("edges", edges_image);
	cvSaveImage("images/stroke_edge_detection.jpg",edges_image);
	printf("edge detection done\n");
	printf("starting text width approximation...\n");
	tester->start_timer();
	int text_width =text_width_approximation(edges_image);
	tester->stop_timer();
	printf("text width: %d",text_width);
	printf("text width approximation done\n");
	printf("starting local thresholding...\n");
	tester->start_timer();
	IplImage * timage = local_thresholding(compensated_img,edges_image,text_width);
	tester->stop_timer();
	printf("local thresholding done\n");
	display_image("tresholded", timage);
	//cvSaveImage("images/local_thresholding.jpg",timage);
	//printf("starting dispekle...\n");
	//tester->start_timer();
	//IplImage * dispekle_img = despeckle(timage,image,back_img);
	//tester->stop_timer();
	//printf("dispekle done.\n");
	//display_image("dispekle", dispekle_img);
	//cvSaveImage("images/dispekle.jpg",dispekle_img);
	//printf("starting logical operators...\n");
	//tester->start_timer();
	//IplImage * finalImg = apply_logical_operators(dispekle_img);
	//tester->stop_timer();
	//printf("logical operators done.\n");
	//display_image("final",finalImg);
	//cvSaveImage("images/finalimage.pgm",finalImg);
	//cvSaveImage("images/finalimage.jpg",finalImg);
	//tester->time_algorithm(;

}
void test_performance(){
	//char * filename= "C:\\Users\\ninao\\Documents\\images\\testimages\\bigimage.bmp";
	//image = cvLoadImage(filename,0);
	Tester* tester = new Tester();
	tester->start_timer();
	IplImage* rimage = lu_algorithm(image);
	tester->stop_timer();
	display_image("rimage",rimage);

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
		IplImage * rimage = lu_algorithm(input_image);
		cvSaveImage(filename_result,rimage);
		cvReleaseImage(&rimage);
	}
	else{
		printf("enter more arguments");
	}
}


int _tmain(int argc, _TCHAR* argv[])
{

	run_as_shell_command(argc, argv);

	//image = cvLoadImage("C:\\Users\\Oliver\\Projects\\FHTW2013\\binatool\\dibco09-images\\H04.bmp",0);
	//IplImage * rimage = lu_algorithm(image);
	//display_image("su",rimage);

	//char * filename= "C:\\Users\\ninao\\Documents\\images\\mediumimage.tif";
	//image = cvLoadImage(filename,0);
	//display_image("original", image);
	//LuAlgorithmTester lutester(image) ;
	//lutester.test_complete_algorithm();
	//test_time_lu_algorithm();
	//test_performance();
	 

	//char * file_to_open = "C:\\Users\\ninao\\Documents\\images\\dibco_test_images\\H04.bmp";
	//char * file_to_save = "C:\\Users\\ninao\\Documents\\images\\results\\lu\\H04.png";
	////
	//Tester * tester = new Tester(file_to_open,file_to_save);
	//tester->time_algorithm(lu_algorithm);

	//tester->time_algorithm(hdibco_algorithm);
	//tester->test_dicta_algorithm_steps();
	//tester->test_batch_thresholding();
	//test_background_estimation();
	
	//cvWaitKey(0);
	return 0;
}

