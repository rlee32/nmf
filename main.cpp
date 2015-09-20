// #include "entities.h"
// #include "matrix.h"
// #include "nmf.h"
#include "debug.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <ctime>

// void tests()
// {
// 	std::cout << "\n\n\n";
// 	test_substitution_cpu();
// 	test_matmult_ata_lowertriangular_cpu();
// 	test_matvecmult_transpose_cpu();
// 	test_cholesky_lowertriangular_cpu();
// 	test_normal_equations_cpu();
// 	test_markInfeasible();
// 	test_nnls_single_cpu();
// 	test_nnls_multiple_cpu_singlerhs();
// 	test_nnls_multiple_cpu();
// 	test_nmf_cpu();
// 	// test_reader("data/arcene_test.data",' ');
// }

int main(int argc, char *argv[])
{
	srand(time(NULL));
	if(argc < 4)
	{
		std::cout << "Input arguments:\n\t1: Filename for A matrix (as in A ~= WH)\n\t2: New desired dimension\n\t3: Max NMF iterations\n\t4: Max NNLS iterations\n\t5 (optional): delimiter (space is default)\n";
	}
	else
	{
		std::string filename = argv[1];
		int newDimension = atoi(argv[2]);
		int max_iter_nmf = atoi(argv[3]);
		int max_iter_nnls = atoi(argv[4]);
		char delimiter = (argc > 5) ? *argv[5] : ' ';

		DenseMatrix* A = readMatrix(filename,delimiter);
		A->copyColumnToRow();
		printf("Sparsity of A: %f\n",sparsity(A));

		DenseMatrix W = DenseMatrix(A->rows,newDimension);
		DenseMatrix H = DenseMatrix(newDimension,A->cols);
		NMF_Input input = NMF_Input(&W,&H,A,max_iter_nmf,max_iter_nnls);

		std::cout << "Starting NMF computation." << std::endl;
		std::clock_t start = std::clock();
		double duration;
		// nmf_cpu(input);
		nmf_cpu_profile(input);
		duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
		std::cout << "NMF computation complete. Time: " << duration << " s." << std::endl;

		W.copyColumnToRow();
		dtype AF = FrobeniusNorm(A);
		dtype WH_AF = Fnorm(W,H,*A);
		printf("Objective value: %f\n",WH_AF/AF);
		// DenseMatrix z1 = DenseMatrix(A->rows,newDimension);
		// DenseMatrix z2 = DenseMatrix(newDimension,A->cols);
		// printf("Calculated solution approximate Frobenius norm: %f\n",Fnorm(z1,z2,*A));
		// printcolmajor(H.colmajor,H.rows,H.cols);
		if(A) delete A;
	}
	// tests();
	return 0;
}