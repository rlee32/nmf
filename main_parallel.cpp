// #include "entities.h"
// #include "matrix.h"
// #include "nmf.h"
#include "debug.h"
#include "debug_gpu.h"

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
	// tests();
	// test_level3();
	// srand(time(NULL));
	srand(10);
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

		std::cout << "Starting GPU preparation..." << std::endl;
		gpu_input gpu_in;
		gpumalloc(gpu_in.d_A_colmajor,A->totalsize);
		gpumalloc(gpu_in.d_W_colmajor,W.totalsize);
		gpumalloc(gpu_in.d_H_colmajor,H.totalsize);
		gpumalloc(gpu_in.d_HAT_colmajor,H.rows*A->rows);
		gpumalloc(gpu_in.d_WTA_colmajor,W.cols*A->cols);
		colmatrix2gpu(gpu_in.d_A_colmajor,A->colmajor,A->cols,A->rows);
		gpusync();
		std::cout << "GPU prep finished." << std::endl;

		std::cout << "Starting NMF computation." << std::endl;
		std::clock_t start = std::clock();
		double duration;
		// nmf_cpu(input);
		nmf_gpu_profile(input,gpu_in);
		// nmf_gpu_profile(input);
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
		gpufree(gpu_in.d_A_colmajor);
		gpufree(gpu_in.d_W_colmajor);
		gpufree(gpu_in.d_H_colmajor);
		gpufree(gpu_in.d_HAT_colmajor);
		gpufree(gpu_in.d_WTA_colmajor);
		if(A) delete A;
	}
	return 0;
}