#pragma once

#include "matrix.h"
#include "common.h"


void test_level3()
{
	int Arows = 20;
	int Acols = 20;
	int Brows = Acols;
	int Bcols = 20;
	int Crows = Arows;
	int Ccols = Bcols;

	dtype**A_colmajor = new dtype*[Acols];
	for(int col = 0; col < Acols; ++col) A_colmajor[col] = new dtype[Arows];
	dtype**B_colmajor = new dtype*[Bcols];
	for(int col = 0; col < Bcols; ++col) B_colmajor[col] = new dtype[Brows];

	randInit(A_colmajor,Arows,Acols,10);
	randInit(B_colmajor,Brows,Bcols,10);

	DenseMatrix A = DenseMatrix(A_colmajor,Arows,Acols);
	DenseMatrix B = DenseMatrix(B_colmajor,Brows,Bcols);
	DenseMatrix C = DenseMatrix(Crows,Ccols);

	matmatmult_colmajor_cpu(A,B,C);

	dtype *d_A,*d_B,*d_C;
	gpumalloc(d_A,A.totalsize);
	gpumalloc(d_B,B.totalsize);
	gpumalloc(d_C,C.totalsize);
	gpusync();

	printf("copying A\n");
	colmatrix2gpu(d_A,A.colmajor,A.cols,A.rows);
	printf("copying B\n");
	colmatrix2gpu(d_B,B.colmajor,B.cols,B.rows);
	gpusync();

	level3_float(d_A,d_B,d_C,Arows,Acols,Brows,Bcols);
	gpusync();

	dtype**C_fromgpu = new dtype*[Ccols]();
	for(int col = 0; col < Ccols; ++col) C_fromgpu[col] = new dtype[Crows]();
	colmatrix2host(C_fromgpu,d_C,C.cols,C.rows);
	gpusync();

	gpufree(d_A);
	gpufree(d_B);
	gpufree(d_C);

	checkMatrix(C_fromgpu,C.colmajor,C.cols,C.rows,"test_level3",1e-10);

	for(int col = 0; col < Ccols; ++col) delete[] C_fromgpu[col];
	delete[] C_fromgpu;
}

#include "nmf_parallel.h"
