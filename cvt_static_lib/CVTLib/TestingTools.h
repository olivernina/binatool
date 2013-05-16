#ifndef TESTING_TOOLS_HEADER
#define TESTING_TOOLS_HEADER

//#define _CRTDBG_MAP_ALLOC
//#include <stdlib.h>
//#include <crtdbg.h>

#include "FunctionsLib.h"
#include <ctime>


class Tester{
	public:
		char * file_to_save;
		char * file_to_open;
		IplImage * image;
		clock_t start;
		clock_t total_time;
		bool first_run;
		Tester();
		Tester(char * file_to_open, char * file_to_save);
		void time_algorithm(IplImage * (*function)(IplImage*));
		void test_batch_thresholding();
		void start_timer();
		void stop_timer();
		void get_total_testing_time();

};


#endif