// Nina.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Nina.h"


IplImage *image;

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
		IplImage * rimage = final_algorithm(input_image);
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

	//char * filename= "C:\\Users\\Oliver\\Projects\\FHTW2013\\binatool\\dibco09-images\\H04.bmp";
	//image = cvLoadImage(filename,0);
	//IplImage * rimage = test_algorithm(image);
	//display_image("lu",rimage);
	//cvWaitKey(0);

	return 0;
}

