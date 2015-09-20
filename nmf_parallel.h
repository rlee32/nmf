
#include <stdio.h>
#include <iostream>
#include "matrix.h"
#include "ls.h"
#include "nnls.h"
#include "nmf.h"
#include "common.h"


void nmf_gpu_profile(NMF_Input& input,gpu_input& gpu_in)
{
	std::clock_t start;
	double allocation_time=0,H_init_time=0,H_copy_time = 0,W_copy_time = 0,HHT_time=0,WTW_time=0,HAT_time=0,WTA_time=0,WT_solve_time=0,H_solve_time=0;

	start = std::clock();
	NMF_State state = NMF_State(input.m,input.k,input.n);
	NNLS_Multiple_Input nnls_input_1 = NNLS_Multiple_Input(state.HHT,input.W->rowmajor,state.HAT,input.m,input.max_iter_nnls);
	NNLS_Multiple_Input nnls_input_2 = NNLS_Multiple_Input(state.WTW,input.H->colmajor,state.WTA,input.n,input.max_iter_nnls);
	// NNLS_Multiple_State nnls_state_1 = NNLS_Multiple_State(nnls_input_1.CTC->dim,nnls_input_1.cols_rhs);
	// NNLS_Multiple_State nnls_state_2 = NNLS_Multiple_State(nnls_input_2.CTC->dim,nnls_input_2.cols_rhs);
	// nnls_input_1.state = &nnls_state_1;
	// nnls_input_2.state = &nnls_state_2;
	allocation_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

	// initializeMatrix(input.H,averageMatrix(input.A));
	start = std::clock();
	initializeMatrix(input.H);
	H_init_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

	int iterations = 0;
	int iterations_nnls = 0;
	while(iterations < input.max_iter_nmf)//TODO: implement solution-sensitive stopping criterion
	{
		start = std::clock();
		input.H->copyColumnToRow();
		H_copy_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

		start = std::clock();
		matmult_ata_lowertriangular_pointers_cpu(*state.HHT,input.H->rowmajor,input.H->cols);
		HHT_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

		start = std::clock();
		// gpucheck("before colmatrix2gpu on H");
	colmatrix2gpu(gpu_in.d_H_colmajor, input.H->colmajor, input.H->cols,input.H->rows);
		// gpusync();
		// gpucheck("before HAT compute");
	level3_HAT_float(gpu_in.d_H_colmajor, gpu_in.d_A_colmajor, gpu_in.d_HAT_colmajor,
		input.H->rows,input.H->cols, input.A->rows,input.A->cols);
		// gpusync();
		// gpucheck("before colmatrix2host on HAT");
	colmatrix2host(state.HAT,gpu_in.d_HAT_colmajor,state.m,state.k);
		// gpusync();
		// int HATrows = input.H->rows;int HATcols = input.A->rows;
		// dtype**gpuHAT = new dtype*[HATcols];
		// for(int col=0;col<HATcols;++col) gpuHAT[col] = new dtype[HATrows]();
		// copy_colmajor(state.HAT,gpuHAT,HATcols,HATrows);
	// for(int i=0;i<input.m;++i) matvecmult_colmajor_cpu(*input.H,input.A->rowmajor[i],state.HAT[i]);
		// check_colmajor(state.HAT,gpuHAT,HATcols,HATrows);
		// for(int col=0;col<HATcols;++col) delete[] gpuHAT[col];
		// delete[] gpuHAT;
	// std::cin.ignore();
		HAT_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

		// printf("H: \n");printr(input.H->rowmajor,input.H->rows,input.H->cols);
		// printc(state.HAT,input.k,input.m);
		// printl(state.HHT);
		// printf("HHT: \n");printl(nnls_input_1.CTC);
		// printf("HAT: \n");printc(nnls_input_1.CTB,input.k,input.m);
		// int iterations1 = 
		start = std::clock();
		iterations_nnls += nnls_multiple_cpu_profile(nnls_input_1);
		WT_solve_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
		// printf("nnls iter 1: %d\n",iterations1);
		// printc(nnls_input_1.X,input.k,input.m);

		start = std::clock();
		input.W->copyRowToColumn();
		W_copy_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

		start = std::clock();
		matmult_ata_lowertriangular_cpu(*state.WTW,*input.W);
		WTW_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

		start = std::clock();
		// gpucheck("before colmatrix2gpu on W");
	colmatrix2gpu(gpu_in.d_W_colmajor,input.W->colmajor,input.W->cols,input.W->rows);
		// gpusync();
		// gpucheck("before WTA compute");
	level3_WTA_float(gpu_in.d_W_colmajor, gpu_in.d_A_colmajor, gpu_in.d_WTA_colmajor,
		input.W->rows,input.W->cols, input.A->rows,input.A->cols);
		// gpusync();
		// gpucheck("before colmatrix2host on WTA");
	colmatrix2host(state.WTA,gpu_in.d_WTA_colmajor,state.n,state.k);
		// gpusync();
		// int WTArows = input.W->cols;int WTAcols = input.A->cols;
		// dtype**gpuWTA = new dtype*[WTAcols];
		// for(int col=0;col<WTAcols;++col) gpuWTA[col] = new dtype[WTArows]();
		// copy_colmajor(state.WTA,gpuWTA,WTAcols,WTArows);
	// for(int i=0;i<input.n;++i) matvecmult_transpose_cpu(*input.W,input.A->colmajor[i],state.WTA[i]);
		// check_colmajor(state.WTA,gpuWTA,WTAcols,WTArows);
		// for(int col=0;col<WTAcols;++col) delete[] gpuWTA[col];
		// delete[] gpuWTA;
	// std::cin.ignore();
		WTA_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
		
		// printf("WTW: \n");printl(nnls_input_2.CTC);
		// printf("WTA: \n");printc(nnls_input_2.CTB,input.k,input.n);
		// int iterations2 = 
		start = std::clock();
		iterations_nnls += nnls_multiple_cpu_profile(nnls_input_2);
		H_solve_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
		// printf("nnls iter 2: %d\n",iterations2);
		// printc(nnls_input_2.X,input.k,input.n);

		++iterations;
	}
	
	printf("\tAverage NNLS Iterations: %f\n",(dtype)iterations_nnls/(dtype)iterations/2.0);
	printf("\tNMF Iterations: %d\n",iterations);

	double total_time = allocation_time+H_init_time+
		H_copy_time+W_copy_time+
		H_solve_time+WT_solve_time+
		HAT_time+WTA_time+
		HHT_time+WTW_time;
	printf("NMF Profiling results (in seconds): (total: %f)\n",total_time);
	printf("\tState and input allocation time: %f (%f %%)\n",allocation_time,allocation_time*100.0/total_time);
	printf("\tH initialization time: %f (%f %%)\n",H_init_time,H_init_time*100.0/total_time);
	printf("\tH, W copy times: %f (%f %%), %f (%f %%)\n",H_copy_time,H_copy_time*100.0/total_time,W_copy_time,W_copy_time*100.0/total_time);
	printf("\tH, WT solve times (nnls): %f (%f %%), %f (%f %%)\n",H_solve_time,H_solve_time*100.0/total_time,WT_solve_time,WT_solve_time*100.0/total_time);
	printf("\tHAT, WTA times (matmult): %f (%f %%), %f (%f %%)\n",HAT_time,HAT_time*100.0/total_time,WTA_time,WTA_time*100.0/total_time);
	printf("\tHHT, WTW times (matmult): %f (%f %%), %f (%f %%)\n",HHT_time,HHT_time*100.0/total_time,WTW_time,WTW_time*100.0/total_time);
	print_nnls_time(nnls_input_1);
	print_nnls_time(nnls_input_2);

}

void nmf_gpu_profile_old(NMF_Input& input,gpu_input& gpu_in)
{
	std::clock_t start;
	double allocation_time=0,H_init_time=0,H_copy_time = 0,W_copy_time = 0,HHT_time=0,WTW_time=0,HAT_time=0,WTA_time=0,WT_solve_time=0,H_solve_time=0;

	start = std::clock();
	NMF_State state = NMF_State(input.m,input.k,input.n);
	NNLS_Multiple_Input nnls_input_1 = NNLS_Multiple_Input(state.HHT,input.W->rowmajor,state.HAT,input.m,input.max_iter_nnls);
	NNLS_Multiple_Input nnls_input_2 = NNLS_Multiple_Input(state.WTW,input.H->colmajor,state.WTA,input.n,input.max_iter_nnls);
	// NNLS_Multiple_State nnls_state_1 = NNLS_Multiple_State(nnls_input_1.CTC->dim,nnls_input_1.cols_rhs);
	// NNLS_Multiple_State nnls_state_2 = NNLS_Multiple_State(nnls_input_2.CTC->dim,nnls_input_2.cols_rhs);
	// nnls_input_1.state = &nnls_state_1;
	// nnls_input_2.state = &nnls_state_2;
	allocation_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

	// initializeMatrix(input.H,averageMatrix(input.A));
	start = std::clock();
	initializeMatrix(input.H);
	H_init_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

	int iterations = 0;
	int iterations_nnls = 0;
	while(iterations < input.max_iter_nmf)//TODO: implement solution-sensitive stopping criterion
	{
		start = std::clock();
		input.H->copyColumnToRow();
		H_copy_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

		start = std::clock();
		matmult_ata_lowertriangular_pointers_cpu(*state.HHT,input.H->rowmajor,input.H->cols);
		HHT_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

		start = std::clock();
		// gpucheck("before colmatrix2gpu on H");
		// colmatrix2gpu(gpu_in.d_H_colmajor, input.H->colmajor, input.H->cols,input.H->rows);
		// gpusync();
		// gpucheck("before HAT compute");
		// level3_HAT_float(gpu_in.d_H_colmajor, gpu_in.d_A_colmajor, gpu_in.d_HAT_colmajor,
		// 	input.H->rows,input.H->cols, input.A->rows,input.A->cols);
		// gpusync();
		// gpucheck("before colmatrix2host on HAT");
		// colmatrix2host(state.HAT,gpu_in.d_HAT_colmajor,state.m,state.k);
		// gpusync();
		// int HATrows = input.H->rows;int HATcols = input.A->rows;
		// dtype**gpuHAT = new dtype*[HATcols];
		// for(int col=0;col<HATcols;++col) gpuHAT[col] = new dtype[HATrows]();
		// copy_colmajor(state.HAT,gpuHAT,HATcols,HATrows);
			for(int i=0;i<input.m;++i) matvecmult_colmajor_cpu(*input.H,input.A->rowmajor[i],state.HAT[i]);
		// check_colmajor(state.HAT,gpuHAT,HATcols,HATrows);
		// for(int col=0;col<HATcols;++col) delete[] gpuHAT[col];
		// delete[] gpuHAT;
	// std::cin.ignore();
		HAT_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

		// printf("H: \n");printr(input.H->rowmajor,input.H->rows,input.H->cols);
		// printc(state.HAT,input.k,input.m);
		// printl(state.HHT);
		// printf("HHT: \n");printl(nnls_input_1.CTC);
		// printf("HAT: \n");printc(nnls_input_1.CTB,input.k,input.m);
		// int iterations1 = 
		start = std::clock();
		// gpusync();
		iterations_nnls += nnls_multiple_cpu_profile(nnls_input_1);
		WT_solve_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
		// printf("nnls iter 1: %d\n",iterations1);
		// printc(nnls_input_1.X,input.k,input.m);

		start = std::clock();
		input.W->copyRowToColumn();
		W_copy_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

		start = std::clock();
		matmult_ata_lowertriangular_cpu(*state.WTW,*input.W);
		WTW_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

		start = std::clock();
		// gpucheck("before colmatrix2gpu on W");
		// colmatrix2gpu(gpu_in.d_W_colmajor,input.W->colmajor,input.W->cols,input.W->rows);
		// gpusync();
		// gpucheck("before WTA compute");
		// level3_WTA_float(gpu_in.d_W_colmajor, gpu_in.d_A_colmajor, gpu_in.d_WTA_colmajor,
		// 	input.W->rows,input.W->cols, input.A->rows,input.A->cols);
		// gpusync();
		// gpucheck("before colmatrix2host on WTA");
		// colmatrix2host(state.WTA,gpu_in.d_WTA_colmajor,state.n,state.k);
		// gpusync();
		// int WTArows = input.W->cols;int WTAcols = input.A->cols;
		// dtype**gpuWTA = new dtype*[WTAcols];
		// for(int col=0;col<WTAcols;++col) gpuWTA[col] = new dtype[WTArows]();
		// copy_colmajor(state.WTA,gpuWTA,WTAcols,WTArows);
			for(int i=0;i<input.n;++i) matvecmult_transpose_cpu(*input.W,input.A->colmajor[i],state.WTA[i]);
		// check_colmajor(state.WTA,gpuWTA,WTAcols,WTArows);
		// for(int col=0;col<WTAcols;++col) delete[] gpuWTA[col];
		// delete[] gpuWTA;
	// std::cin.ignore();
		WTA_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

		
		// printf("WTW: \n");printl(nnls_input_2.CTC);
		// printf("WTA: \n");printc(nnls_input_2.CTB,input.k,input.n);
		// int iterations2 = 
		start = std::clock();
		// gpusync();
		iterations_nnls += nnls_multiple_cpu_profile(nnls_input_2);
		H_solve_time += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
		// printf("nnls iter 2: %d\n",iterations2);
		// printc(nnls_input_2.X,input.k,input.n);

		++iterations;
	}
	
	printf("\tAverage NNLS Iterations: %f\n",(dtype)iterations_nnls/(dtype)iterations/2.0);
	printf("\tNMF Iterations: %d\n",iterations);

	double total_time = allocation_time+H_init_time+
		H_copy_time+W_copy_time+
		H_solve_time+WT_solve_time+
		HAT_time+WTA_time+
		HHT_time+WTW_time;
	printf("NMF Profiling results (in seconds): (total: %f)\n",total_time);
	printf("\tState and input allocation time: %f (%f %%)\n",allocation_time,allocation_time*100.0/total_time);
	printf("\tH initialization time: %f (%f %%)\n",H_init_time,H_init_time*100.0/total_time);
	printf("\tH, W copy times: %f (%f %%), %f (%f %%)\n",H_copy_time,H_copy_time*100.0/total_time,W_copy_time,W_copy_time*100.0/total_time);
	printf("\tH, WT solve times (nnls): %f (%f %%), %f (%f %%)\n",H_solve_time,H_solve_time*100.0/total_time,WT_solve_time,WT_solve_time*100.0/total_time);
	printf("\tHAT, WTA times (matmult): %f (%f %%), %f (%f %%)\n",HAT_time,HAT_time*100.0/total_time,WTA_time,WTA_time*100.0/total_time);
	printf("\tHHT, WTW times (matmult): %f (%f %%), %f (%f %%)\n",HHT_time,HHT_time*100.0/total_time,WTW_time,WTW_time*100.0/total_time);
	print_nnls_time(nnls_input_1);
	print_nnls_time(nnls_input_2);

}