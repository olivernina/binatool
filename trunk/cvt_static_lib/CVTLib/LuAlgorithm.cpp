#include "LuAlgorithm.h"
#include "FunctionsLib.h"
#include "math.h"



using namespace cvt;

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

//float get_median(vector<int> list){
//	if(list.size()==1)
//		return (float)list.at(0);
//
//	sort (list.begin(), list.end(), sorting_function);
//	int size = list.size();
//	int middle = size/2;
//	float median;
//	if (size%2==0) 
//		median = (float)(list.at(middle-1)+list.at(middle))/2;
//	else 
//		median = (float)(list.at(middle));
//
//	return median;
//}


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
	IplImage * rimage = cvCloneImage(image);
	//display_image("rimage",rimage);
	int ratio = 10; 
	int iter_limit = image->width/(2*ks*ratio); //this variable has been deprecated
	for(int i=0;i<image->height;i++){
		vector<int> signal = get_row_image(image,i);
		vector<Coordinates> sampled_signal = sample_signal(signal, ks);
		vector<int> values= iterative_fitting(signal,sampled_signal, iter_limit);
        rimage = paint_row(rimage,values,i);
        
	}
	iter_limit = image->height/(2*ks*ratio);
	for(int i=0;i<image->width;i++){
		vector<int> signal = get_col_image(rimage,i);
		vector<Coordinates> sampled_signal = sample_signal(signal, ks);
		vector<int> values= iterative_fitting(signal,sampled_signal,iter_limit);
        rimage = paint_col(rimage,values,i);
	}
     
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

IplImage * edge_detection(IplImage * image){
	IplImage * rimage = cvCloneImage(image);
    cvSetZero(rimage);
	int array_size = 256;
    int * histogram = new int[array_size];
    clear_histogram(histogram, array_size);
	//print_array(histogram,array_size);
	int max_value = get_max_value_edge(image);
	for(int i=1; i<image->height-1;i++){
		for(int j=1;j<image->width-1;j++){
			int pixel_left = get_pixel(image,i,j-1);
			int pixel_right = get_pixel(image,i,j+1);
			int V_h = (int)fabs((float)pixel_left - pixel_right);
			int pixel_up = get_pixel(image,i-1,j);
			int pixel_down = get_pixel(image,i+1,j);
            int V_v = (int)fabs((float) pixel_up - pixel_down);
            int V  = V_h + V_v;
			V = (float)255/max_value * V;
            //cvSet2D(rimage,i,j,cvScalar(V,V,V));
			set_pixel(rimage,i,j,V);
			if(V>255){
				printf("out of range");
				continue;
			}
            histogram[V]++;
		}
	}

    int threshold1 = fast_otsu_algorithm(histogram);
	//int threshold2 = kittler_threshold_histogram(histogram);
	//printf("otsu: %d",threshold1);
	//printf("kittler: %d",threshold2);
	//float thresh = ((float)threshold1 + (float)threshold2)/2.0 ;
	//printf("thresh: %f",thresh);
    rimage = apply_threshold(rimage,threshold1,1);
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

void get_distance_next_stroke(uchar* row,int * histogram, int hist_size,int img_width){
	for(int index=0;index<img_width;index++){
		uchar pixel =row[index];
		int edge_value = 255;
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


int text_width_approximation(IplImage * image){

	int histSize = 36;//This is an assumption. This should be carefully examined
    int * histogram = create_histogram(histSize); 
	for(int i=0;i<image->height;i++){
        uchar* row = get_row_data(image,i);
		get_distance_next_stroke(row,histogram,histSize,image->width);
	}
	//print_array(histogram,histSize);
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
					//cvSet2D(rimage,j,i,cvScalar(255,255,255));
					set_pixel(rimage,j,i,255);
					continue;
			}

            int Ne = get_number_edge_pixels(edges_img,i,j,radius);
            int Emean = get_edges_mean(edges_img,compensate_img,i,j,radius);
			int pixel_comp = get_pixel(compensate_img,j,i);
			if( Ne >= Nmin && pixel_comp <= Emean){
				//cvSet2D(rimage,j,i,cvScalar(0,0,0));
				set_pixel(rimage,j,i,0);
			}
			else{
				//cvSet2D(rimage,j,i,cvScalar(255,255,255));
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

	#ifdef GSL
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
	#endif
	return true; /* we do not "analyse" the result (cov matrix mainly)
				 to know if the fit is "good" */

}

	IplImage * lu_algorithm(IplImage * image,int ks){
		
		//printf("starting background estimation...\n");
		IplImage * back_img = background_estimation(image,ks);
		//cvSaveImage("results/background.png",back_img);
		//IplImage * back_img = iterative_background_estimation(image);
		//printf("background estimation done\n");
		//printf("starting ccv...\n");

		//IplImage* backsub = cvCloneImage(back_img);
		//cvSub(back_img,image,backsub);
		//backsub = invert_image(backsub);
		//cvShowImage("backsub",backsub);

		//IplImage *compensated_img = compensate_contrast_variation_normalized(image,back_img);
		IplImage *compensated_img = compensate_contrast_variation(image,back_img);

		//cvSaveImage("results/comp.png",compensated_img);
		//printf("ccv completed\n");
		//printf("starting background estimation...\n");
		//printf("starting edge detection...\n");

		IplImage * edges_image = edge_detection(compensated_img);
		//IplImage * edges_image = cvCloneImage(image);
		//cvCanny(compensated_img,edges_image,50, 200);
		//cvSaveImage("results/edges.png",edges_image);
		//printf("edge detection done\n");
		//printf("starting text width approximation...\n");
		int text_width =text_width_approximation(edges_image);
		
		//printf("text width: %d",text_width);
		//printf("text width approximation done\n");
		//printf("starting local thresholding...\n");
		IplImage * timage = local_thresholding(compensated_img,edges_image,text_width);
		//cvSaveImage("results/timage.png",timage);
		//printf("local thresholding done\n");
		//printf("starting dispekle...\n");

		IplImage * dispekle_img = despeckle(timage,image,back_img);
		//cvSaveImage("results/dispekle.png",dispekle_img);
		////printf("dispekle done.\n");
		////printf("starting logical operators...\n");
		IplImage * final_img = apply_logical_operators(dispekle_img);
		//cvSaveImage("results/final.png",final_img);
		////printf("logical operators done.\n");
		return final_img;
		//return timage;
//		return edges_image;
	}