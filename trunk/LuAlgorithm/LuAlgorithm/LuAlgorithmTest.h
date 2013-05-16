#ifndef LU_ALGORITHM_TEST_HEADER
#define LU_ALGORITHM_TEST_HEADER

#include "FunctionsLib.h"
class LuAlgorithmTester {
	public: 
		LuAlgorithmTester(IplImage * image);
		void test_complete_algorithm();
		void test_polyfit();
		
};

#endif