#include "Nina.h"


void display_image(char * winName,IplImage* image){
	cvNamedWindow(winName,0);
	cvShowImage(winName,image);
}
int get_pixel(IplImage * image, int row,int col){ //row,col
		int value = ((uchar*)(image->imageData + image->widthStep*row))[col];
		return value;
}
void set_pixel(IplImage * image, int row,int col, int new_value){ 
	((uchar*)(image->imageData + image->widthStep*row))[col]= (uchar)new_value;
}
bool sorting_function (int i,int j) { return (i<j); }

vector<int> get_row_image(IplImage * image,int i){

	vector<int> signal;
	for(int j =0;j<image->width;j++){
		int value = get_pixel(image,i,j);
		signal.push_back(value);
		//printf("%d",signal[j]);    
	}
	return signal;
	
}
vector<int> get_col_image(IplImage * image,int j){

	vector<int> signal;
	for(int i =0;i<image->height;i++){
		int value =get_pixel(image,i,j);
		signal.push_back(value);
		//printf("%d",signal[j]);    
	}
	return signal;
	
}
IplImage * paint_row(IplImage * image,vector<int> values,int row){
	for(int col=0;col<values.size();col++){
		int value = values.at(col);
        /*cvSet2D(image,row,col,cvScalar(value,value,value));*/
		set_pixel(image,row,col,value);
	}
    return image;
}
IplImage * paint_col(IplImage * image,vector<int> values,int col){
	for(int row=0;row<values.size();row++){
		int value = values.at(row);
        cvSet2D(image,row,col,cvScalar(value,value,value));
	}
    return image;
}
vector<int> get_sub_signal(vector<int> signal,int ipos,int fpos){
	vector<int> sub_signal;
	for(int i = ipos;i<fpos;i++){
		sub_signal.push_back(signal.at(i));
	}
	return sub_signal;
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


vector<Coordinates> sample_signal(vector<int> signal, int ks){ 

	vector<Coordinates> sampled_signal;
	int final_iter = (int)floor((float)signal.size()/ks); 

	for(int i=0;i<final_iter;i++){
		float X = ks * i;
		float init_val = X-.5*ks;
		int I_o = 0 ;
		if(init_val > 0)
			I_o = (int)ceil(init_val);

		int I_i = (int)ceil(X+.5*ks);
		
		
		vector<int> templist = get_sub_signal(signal,I_o,I_i);

		if(templist.size()>0){ 
			int y = floor(get_median(templist));
			struct Coordinates coord;
			coord.x = X;
			coord.y = y;
			sampled_signal.push_back(coord);
		}
		//printf("%d",i);
	}
	return sampled_signal;
}

double * polynomial_fitting(vector<Coordinates> signal,int degree){
	degree = degree+1;
	double* coeffs = new double[degree];
	int obs = signal.size();
	double * x = new double[obs];
	double* y = new double[obs];

	for(int i=0;i<obs;i++){
		x[i] = signal.at(i).x;
		y[i] = signal.at(i).y;
	}

	polyfit(obs, degree, x, y, coeffs);
	return coeffs;
}

int get_y_coordinate(int x,double* B,int degree){
	double sum = 0;
	for(int d=0;d<=degree;d++){
		double b = B[d];
        sum = sum + (b* pow((double)x,d));
	}
	int result = (int)ceil(sum);
    return result;
}
vector<int> calculate_fit_polynomial(vector<int> signal,double * coeffs,int degree){
	vector<int> y;

	for(int i=0;i<signal.size();i++){
		int value = get_y_coordinate(i,coeffs,degree);
		y.push_back(value);

	}
	return y;
}
Coordinates get_max_error(vector<Coordinates> sampled_signal, vector<int>  fitted_signal){
	int max_error = 0;
    int pos = -1;
	for(int i=0; i< sampled_signal.size();i++){
		Coordinates  sample= sampled_signal.at(i);
		int x = sample.x;
		int y = sample.y;
        int error = abs(y-fitted_signal.at(x));
		if(error > max_error){
            max_error =error ;
            pos = i;
		}
	}
	Coordinates tuple;
	tuple.x = max_error;
	tuple.y = pos;
    return  tuple;
}
vector<int> iterative_fitting(vector<int>signal,vector<Coordinates> sampled_signal, int iter_limit){
	int n=1;
    vector<int> fitted_polynomial;
    double * coeffs;
	int d_i = 0;
	while(true){
        float kt = .1;
        float d_o = 6;
        d_i = d_o + floor(kt * n);
		//printf(" d_i: %d ",d_i);
        coeffs = polynomial_fitting(sampled_signal,d_i);
		//print_array(coeffs,d_i+1);
        fitted_polynomial = calculate_fit_polynomial(signal,coeffs,d_i);
		//print_array(fitted_polynomial,signal.size());
		Coordinates max_error = get_max_error(sampled_signal, fitted_polynomial);
		//printf("error: %d  pos: %d",max_error.x,max_error.y);
		if(max_error.x > 10 &&  sampled_signal.size() < d_i ){ 
			sampled_signal.erase(sampled_signal.begin()+max_error.y);

		}
		else{
            break;
		}
        n++;
	}
    
    fitted_polynomial = calculate_fit_polynomial(signal, coeffs,d_i);// fitted polynomial of original signal
    return fitted_polynomial;
}

IplImage * background_estimation(IplImage * image, int ks){

	IplImage * polynomial_image = cvCloneImage(image);
	IplImage * median_image = cvCloneImage(image);

	cvSmooth(image,median_image,CV_MEDIAN,21,21);
	int ratio = 10; 
	int iter_limit = image->width/(2*ks*ratio); 

	for(int i=0;i<image->height;i++){
		vector<int> signal = get_row_image(image,i);
		vector<Coordinates> sampled_signal = sample_signal(signal, ks);
		vector<int> values= iterative_fitting(signal,sampled_signal, iter_limit);
        polynomial_image = paint_row(polynomial_image,values,i);
        
	}

	iter_limit = image->height/(2*ks*ratio);
	for(int i=0;i<image->width;i++){
		vector<int> signal = get_col_image(polynomial_image,i);
		vector<Coordinates> sampled_signal = sample_signal(signal, ks);
		vector<int> values= iterative_fitting(signal,sampled_signal,iter_limit);
        polynomial_image = paint_col(polynomial_image,values,i);
	}
     
	IplImage * rimage = cvCloneImage(image);
	for(int i=0;i<image->height;i++)
		for(int j=0;j<image->width;j++)
		{
			int pixel = get_pixel(polynomial_image,i,j);
			int pixel2 = get_pixel(median_image,i,j);
	
			//if(pixel > pixel2)
			//	set_pixel(rimage3,i,j,pixel);
			//else
			//	set_pixel(rimage3,i,j,pixel2);

			set_pixel(rimage,i,j,(pixel + pixel2)/2.);

		}

	cvReleaseImage(&polynomial_image);
	cvReleaseImage(&median_image);

	return rimage;
}

void print_vector(vector<int>y){
	for(int i=0; i < y.size(); i++) {
		printf(" %ld ", y.at(i));
	}
}

void print_array(double * a,int size){
	for(int i=0; i < size; i++) {
		printf(" %lf ", a[i]);
	}
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
int calculate_mean(IplImage *image){
	int values_sum = 0;
	int num_pixels=0;
	for(int i=0; i<image->height;i++){
		for(int j=0;j<image->width;j++){
			int value = get_pixel(image,i,j);
			values_sum += value; 
			num_pixels++;
		}
	}
	float mean = (float)values_sum/num_pixels;
	int result = ceil(mean);
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

int get_max_value_edge(IplImage * image){
	int max_value = 0;
	for(int i=1; i<image->height-1;i++){
		for(int j=1;j<image->width-1;j++){
			int V_h = fabs((float)get_pixel(image,i,j-1) - get_pixel(image,i,j+1));
            int V_v = fabs((float)get_pixel(image,i-1,j) - get_pixel(image,i+1,j));
            int V  = V_h + V_v;
			if(V>max_value)
				max_value = V;
		}
	}
	return max_value;
}

int otsu_algorithm(int* histogram) {
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
IplImage * apply_threshold(IplImage* image,int threshold, int type){  //type -1 means sign is less than, 1 is greater than, default value =-1
	IplImage* rimage = cvCloneImage(image);

	for(int y = 0; y < image->height; y++){
		for(int x = 0; x < image->width; x++){
			int pixel =get_pixel(image,y,x);
			if(type>0){
				if(pixel>threshold){
					set_pixel(rimage,y,x,0);
				}
				else{
					set_pixel(rimage,y,x,255);
				}
			}
			if(type<0){  
				if(pixel<threshold){// add the equal sign here to increase recall percentile this is a bug that will make dicta_algorithm work
					set_pixel(rimage,y,x,0);
				}
				else{
					set_pixel(rimage,y,x,255);
				}
			}

		}		  
	}
	return rimage;
}
void clear_histogram(int * histogram,int size){
	memset(histogram,0,size*sizeof(int));//You could do it this way instead
	//for(int i=0;i<size;i++){
	//	histogram[i] = 0;
	//}

}

uchar* get_row_data(IplImage * image,int row){ // sinonym of the getRowData but with a different format coding
	uchar* row_data = ((uchar*)(image->imageData + image->widthStep*row));
	return row_data;
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

IplImage * otsu_algorithm(IplImage * image){
	int * hist = get_histogram(image);
	float T1 = otsu_threshold_histogram(hist);
	float T2 = kittler_threshold_histogram(hist);

	float T = (T1+T2)/2.;

	IplImage * thresholded_img = apply_threshold(image,T);
	return thresholded_img;
}


IplImage * kittler_algorithm(IplImage * image){
	int * hist = get_histogram(image);
	int T = kittler_threshold_histogram(hist);
	IplImage * thresholded_img = apply_threshold(image,T);
	return thresholded_img;
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
	cvReleaseStructuringElement(&kernel);
	return dst_image;

}


#define INF 1e10 
void normalizeImage(IplImage * input, IplImage * output) {

    assert ( input->depth == IPL_DEPTH_32F );
    assert ( input->nChannels == 1 );
    assert ( output->depth == IPL_DEPTH_32F );
    assert ( output->nChannels == 1 );
    float maxVal = 0;
    float minVal = INF;
    for( int row = 0; row < input->height; row++ ){
        const float* ptr = (const float*)(input->imageData + row * input->widthStep);
        for ( int col = 0; col < input->width; col++ ){
            if (*ptr == INF) { }
            else {
                maxVal = std::max(*ptr, maxVal);
                minVal = std::min(*ptr, minVal);
            }
            ptr++;
        }
    }

    float difference = maxVal - minVal;
    for( int row = 0; row < input->height; row++ ){
        const float* ptrin = (const float*)(input->imageData + row * input->widthStep);\
        float* ptrout = (float*)(output->imageData + row * output->widthStep);\
        for ( int col = 0; col < input->width; col++ ){
            if (*ptrin == INF) {
                *ptrout = 1;
            } else {
                *ptrout = ((*ptrin) - minVal)/difference;
            }
            ptrout++;
            ptrin++;
        }
    }
}

IplImage * normalizeToGray(IplImage* image){
	IplImage * output_normalized = cvCreateImage ( cvGetSize ( image ), IPL_DEPTH_32F, 1 );
	normalizeImage(image, output_normalized);
	IplImage * gray_output = cvCreateImage ( cvGetSize ( image ), IPL_DEPTH_8U, 1 );
	cvConvertScale(output_normalized, gray_output, 255, 0);
	cvReleaseImage(&output_normalized);
	return gray_output;
}

IplImage * edge_detection(IplImage * image){

	int kernel_size =5 ;
	CvSize image_size = cvGetSize ( image );
	IplImage * dx = cvCreateImage ( image_size, IPL_DEPTH_32F, 1 );
	IplImage * dy = cvCreateImage (image_size, IPL_DEPTH_32F, 1 ); 
	IplImage* smoothed_image = cvCloneImage(image);
	cvSmooth( image, smoothed_image, CV_GAUSSIAN, kernel_size,kernel_size);

	cvSobel(smoothed_image, dx , 1, 0, 3);
	cvSobel(smoothed_image, dy , 0, 1, 3);    



	IplImage * magnitude = cvCreateImage(image_size,IPL_DEPTH_32F,1);

	for(int i = 0; i < image->height; i++)
	{
		for(int j = 0; j < image->width; j++)
		{

			float y = CV_IMAGE_ELEM(dy, float,i,j);
			float x = CV_IMAGE_ELEM(dx, float,i,j);
			float magval = abs(x) + abs(y);
			CV_IMAGE_ELEM(magnitude,float,i,j) = magval; 
		}
	}

	IplImage * nmagnitude= normalizeToGray(magnitude);
	IplImage * otsu = otsu_algorithm(nmagnitude);
	IplImage * rimage = invert_image(otsu);

	cvReleaseImage(&dx);
	cvReleaseImage(&dy);
	cvReleaseImage(&smoothed_image);
	cvReleaseImage(&magnitude);
	cvReleaseImage(&otsu);
	cvReleaseImage(&nmagnitude);

	return rimage;
}

int find_next_edge_pixel(vector<int>row,int index){
    int next_index = index; //in case I don't find a next pixel stroke
	for(int i=index+1;i<row.size();i++){
		if(row[i]== 0){
            int next_index = i;
            return next_index;
		}
	}
    return next_index;
}

int argmax(int * histogram,int size){
	int max = 0;
	int max_pos = 0;
	for(int i=0;i<size;i++){
		if(histogram[i]>max){
			max = histogram[i];
			max_pos = i;
		}

	}
	return max_pos;
}

void get_distance_next_stroke(vector<int> row,int * histogram){
	for(int index=0;index<row.size();index++){
		int distance = 0;
		int pixel =row.at(index);
		if(pixel == 0){
			int start_index = index;
			int end_index = find_next_edge_pixel(row,index);
			int distance = end_index - start_index - 1;
			if(distance > 2 && distance < 36){
				histogram[distance]++;
			}
			index = end_index;
		}
	}
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

void get_distance_next_stroke(uchar* row,int * histogram, int hist_size,int img_width){
	for(int index=0;index<img_width;index++){
		uchar pixel =row[index];
		int edge_value = 0;
		if(pixel == edge_value){
			int start_index = index;
			int end_index = find_next_edge_pixel(row,index,edge_value,img_width);
			int distance = end_index - start_index - 1;
			if(distance > 2 && distance < hist_size){
				histogram[distance]++;
			}
			index = end_index;
		}
	}
}

int * create_histogram(int size){
	int* hist = (int*)malloc(sizeof(int)*size);
	memset(hist, 0, sizeof(int)*size);
	return hist;
}



int text_width_approximation(IplImage * image){

	int histSize = 36;//This is an assumption. This should be carefully examined
    int * histogram = create_histogram(histSize); 
	for(int i=0;i<image->height;i++){
        uchar* row = get_row_data(image,i);
		get_distance_next_stroke(row,histogram,histSize,image->width);
	}
    int textWidth = argmax(histogram,histSize);
	free(histogram);
    return textWidth;
}

int get_number_edge_pixels(IplImage* edges_img,int i,int j,int radius){
	int kernel_width = i+ radius;
	int kernel_height =  j+ radius;
	int Ne = 0;
	for(int x =i-radius;x < kernel_width+1;x++){
		for(int y=j-radius;y<kernel_height+1;y++){
			int pixel = get_pixel(edges_img,y,x);
			if(pixel ==0)
				Ne++;
		}
	}
	return Ne;
}

float get_edges_mean(IplImage * edges_img,IplImage *image,int i,int j,int radius){
    int kernel_width = i+ radius;
    int kernel_height =  j+ radius;
    float Ne = 0;
    float Esum = 0;
    float Emean = 0;

		for( int x=i-radius;x< kernel_width+1;x++){
			for(int y=j-radius;y< kernel_height+1;y++){
				int edge_pixel = get_pixel(edges_img,y,x);
				if(edge_pixel == 0){
					int pixel = get_pixel(image,y,x);
					Esum = Esum + pixel;
					Ne++;
				}
			}
		}
    if(Ne > 0)
        Emean = Esum/Ne;
    return Emean;
}

IplImage* local_thresholding(IplImage * compensate_img, IplImage * edges_img,int text_width){
    int window_size = text_width * 4;
    if(window_size%2==0) // to make it odd
        window_size++;
    
    int radius = floor((float)window_size /2);
    
    int Nmin = 2 * window_size;
    IplImage * rimage = cvCloneImage(compensate_img);
	int height_limit  =compensate_img->height-radius;
	int width_limit = compensate_img->width-radius;
	for(int j=0; j< compensate_img->height ;j++){ //change radius to 130 ie to debug windows size
		for(int i=0 ; i < compensate_img->width;i++){
			if(j <radius || i < radius || j>= height_limit ||i >= width_limit){ //This is to make whie the borders, we assume there is no text in the borders
					set_pixel(rimage,j,i,255);
					continue;
			}

            int Ne = get_number_edge_pixels(edges_img,i,j,radius);
            int Emean = get_edges_mean(edges_img,compensate_img,i,j,radius);
			int pixel_comp = get_pixel(compensate_img,j,i);
			if( Ne >= Nmin && pixel_comp <= Emean){
				set_pixel(rimage,j,i,0);
			}
			else{
				set_pixel(rimage,j,i,255);
			}
		}
	}
    
    return rimage;
}


Coordinates get_patch_difference2(IplImage * temp_image,IplImage * compensated_img, IplImage * back_img,int color=0){
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
		Coordinates tuple;
		tuple.x= result;
		tuple.y = num_pixels;
		return tuple;
}

int get_num_neighbors(IplImage* image, int y, int x){
	int radius = 1;
	int total = 0;
	for(int i= y-radius; i<= y + radius;i++){
		for( int j = x - radius; j<= x + radius; j++){
			int pixel = get_pixel(image, i,j);
			if(pixel==0)
				total++;
		}
	}
	return total;
}

IplImage * apply_logical_operators(IplImage * dispekled_img){
	IplImage * rimage = cvCloneImage(dispekled_img);
	for(int i = 1; i< dispekled_img->height-1;i++){
		for(int j=1; j < dispekled_img->width-1; j++){
			int num_neighbors = get_num_neighbors(dispekled_img,i,j);
			if(num_neighbors >= 5)
				cvSet2D(rimage,i,j,cvScalar(0,0,0));
			else
				cvSet2D(rimage,i,j,cvScalar(255,255,255));
		}
	}
	return rimage;
}

bool polyfit(int obs, int degree,double *dx, double *dy, double *store) /* n, p */
{

	gsl_multifit_linear_workspace *ws;
	gsl_matrix *cov, *X;
	gsl_vector *y, *c;
	double chisq;

	int i, j;

	X = gsl_matrix_alloc(obs, degree);
	y = gsl_vector_alloc(obs);
	c = gsl_vector_alloc(degree);
	cov = gsl_matrix_alloc(degree, degree);

	for(i=0; i < obs; i++) {
		gsl_matrix_set(X, i, 0, 1.0);
		for(j=0; j < degree; j++) {
			gsl_matrix_set(X, i, j, pow(dx[i], j));
		}
		gsl_vector_set(y, i, dy[i]);
	}

	ws = gsl_multifit_linear_alloc(obs, degree);
	gsl_multifit_linear(X, y, c, cov, &chisq, ws);

	/* store result ... */
	for(i=0,j=degree-1; i < degree; i++,j--)
	{
		store[i] = gsl_vector_get(c, i);
	}

	gsl_multifit_linear_free(ws);
	gsl_matrix_free(X);
	gsl_matrix_free(cov);
	gsl_vector_free(y);
	gsl_vector_free(c);

	return true; /* we do not "analyse" the result (cov matrix mainly)
				 to know if the fit is "good" */

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
			set_pixel(rimage,i,j,I_hat);
		}
	}
	return rimage;

}
int otsu_threshold_histogram(int *hist){ //TODO: assumes the lenght of the histogram is always 256. Otherwise it will fail
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

IplImage * despeckle_threshold(IplImage* timage,IplImage * compensated_img,IplImage * back_img,int threshold){
	float k = .65; // variable suggested to be used by the author upon emailing him   
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
	cvReleaseImage(&tempimage);
	return rimage;
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
	float T = otsu_threshold_histogram(hist);
	rimage = despeckle_threshold(timage,image,back_img,T);
	cvReleaseImage(&tempimage);
	free(hist);
	return rimage;
}
IplImage * test_algorithm(IplImage * image,int ks){
	
	//cvSaveImage("results/original.png",image);
	IplImage * back_img = background_estimation(image,ks);
	//cvSaveImage("results/background.png",back_img);
	IplImage *compensated_img = compensate_contrast_variation(image,back_img);
	//cvSaveImage("results/comp.png",compensated_img);
	IplImage * edges_image = edge_detection(compensated_img);	
	//cvSaveImage("results/edges.png",edges_image);
	int text_width =text_width_approximation(edges_image);
	//printf("%d",text_width);
	IplImage * timage = local_thresholding(compensated_img,edges_image,text_width);
	//cvSaveImage("results/timage.png",timage);
	IplImage * dispekle_img = despeckle(timage,image,back_img);
	//cvSaveImage("results/dispekle.png",dispekle_img);
	IplImage * final_img = apply_logical_operators(dispekle_img);
	//cvSaveImage("results/final.png",final_img);

	cvReleaseImage(&back_img);
	cvReleaseImage(&compensated_img);
	cvReleaseImage(&edges_image);
	cvReleaseImage(&timage);
	cvReleaseImage(&dispekle_img);

	return final_img;
}

IplImage * final_algorithm(IplImage * image,int ks){
	
	IplImage * back_img = background_estimation(image,ks);
	IplImage *compensated_img = compensate_contrast_variation(image,back_img);
	IplImage * edges_image = edge_detection(compensated_img);	
	int text_width =text_width_approximation(edges_image);
	IplImage * timage = local_thresholding(compensated_img,edges_image,text_width);
	IplImage * dispekle_img = despeckle(timage,image,back_img);
	IplImage * final_img = apply_logical_operators(dispekle_img);

	cvReleaseImage(&back_img);
	cvReleaseImage(&compensated_img);
	cvReleaseImage(&edges_image);
	cvReleaseImage(&timage);
	cvReleaseImage(&dispekle_img);

	return final_img;
}


