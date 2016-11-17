#include <iostream>
#include <armadillo>


using namespace std;
using namespace arma;

void KLT()
{
	wall_clock timer;
  	const uint16_t N = 512;
  	const uint16_t k = 480;

  	vec eigval;
  	mat eigvec;

  	mat trans_mat;
  	mat temp = zeros(N,N);

  	//mat A = randi<mat>(N,N, distr_param(0,+255));
  	mat A;
  	A.load("mandrill.pgm",pgm_binary); 

  	/*--------------- Karhunen-Loeve Transform---------------*/
  	timer.tic();

  	// Compute the Mean
  	mat mu = mean(A);

  	//Compute the Covariance
  	mat C = cov(A);

  	//Compute the Eigen values
  	eig_sym(eigval,eigvec, C); //Eigen values in Ascending Order :(

  	// Get the k eigenvectors with highest eigen values
  	trans_mat = eigvec.tail_rows(k);

  	mat::iterator l = mu.begin();
  	for (int i = 0; i < temp.n_rows; i++)
  	{
  		for (int j = 0; j < temp.n_cols; j++)
  		{
  			temp(i,j) = A(i,j) - *l;
  		}
  		++l;
  	}

  	mat Y = trans_mat*temp; //KL Transformation

  	/*---------------------Reconstruction---------------------*/
 	mat re_A;

	re_A = trans_mat.t() * Y;

	l = mu.begin();

	for(int i=0; i< re_A.n_rows; ++i)
	{
		for(int j = 0; j<re_A.n_cols;++j)
		{
			re_A(i,j) += *l;
		}
		++l;
	}  	

	// Done!
  	double n = timer.toc();
  	cout /*<< "\n Original Input\n"    <<A
  		 << "\n Mean\n"              << mu 
  		 << "\n Variance\n"          << temp 
  		 << "\n Result\n"            << Y
  		 << "\n Reconstructed Data\n"<<re_A*/ 
  		 << "\nDone!! Time Elapsed : "      <<n<<" Seconds"<<endl;
  	Y.save("KL_transform.csv",csv_ascii);
  	//A.save("Input.pgm", pgm_binary);
  	re_A.save("Output.pgm", pgm_binary);
  
}

//Wrapper Function
int main()
{
  	KLT();	
  	return 0;
}