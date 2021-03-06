// Intrinsics.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <intrin.h>

#include <memory>
#include <iostream>
#include <vector>

#include <chrono>

#include "matrix.h"

std::shared_ptr<matrix> _m1;
std::shared_ptr<matrix> _m2;

int main()
{
	_m1 = std::make_shared<matrix>(1100, 2300);
	_m2 = std::make_shared<matrix>(2300, 1100);

	{
		// Populate the data with random values
		matrix* m1 = _m1.get();
		matrix* m2 = _m2.get();
		float val(0);
		for (unsigned int row = 0; row < m1->rows(); row++) {
			for (unsigned int col = 0; col < m1->cols(); ++col) {
				m1->setValue(row, col, sin(-val));

				val++;
			}
		}

		val = 0;
		for (unsigned int row = 0; row < m2->rows(); row++) {
			for (unsigned int col = 0; col < m2->cols(); ++col) {
				m2->setValue(row, col, cos(val++));
			}
		}
	}

	std::cout << "m1: " << _m1->rows() << "x" << _m1->cols() << std::endl;
	std::cout << "m2: " << _m2->rows() << "x" << _m2->cols() << std::endl;

	/**** Additions *****/
	{
		std::cout << "Additions:" << std::endl;

		auto startTime = std::chrono::high_resolution_clock::now();
		auto additionNonVectorised = matrix::addNonVectorised(_m1.get(), _m1.get());
		auto endTime = std::chrono::high_resolution_clock::now();
		auto dif = endTime - startTime;

		std::cout << "Non-vectorised Time: " <<
			std::chrono::duration_cast<std::chrono::nanoseconds>(dif).count() <<
			"ns" << std::endl;

		startTime = std::chrono::high_resolution_clock::now();
		auto additionVectorised = matrix::addVectorised256(_m1.get(), _m1.get());
		endTime = std::chrono::high_resolution_clock::now();
		dif = endTime - startTime;

		std::cout << "Vectorised Time:     " <<
			std::chrono::duration_cast<std::chrono::nanoseconds>(dif).count() <<
			"ns" << std::endl;

		std::cout << std::endl;

		if (additionNonVectorised != additionVectorised)
			std::cout << "Differences" << std::endl;
	}

	/**** Subtractiosn *****/
	{
		std::cout << "Subtractions:" << std::endl;

		auto startTime = std::chrono::high_resolution_clock::now();
		auto subtractionNonVectorised = matrix::subtractNonVectorised(_m1.get(), _m1.get());
		auto endTime = std::chrono::high_resolution_clock::now();
		auto dif = endTime - startTime;

		std::cout << "Non-vectorised Time: " <<
			std::chrono::duration_cast<std::chrono::nanoseconds>(dif).count() <<
			"ns" << std::endl;

		startTime = std::chrono::high_resolution_clock::now();
		auto subtractionVectorised = matrix::subtractVectorised256(_m1.get(), _m1.get());
		endTime = std::chrono::high_resolution_clock::now();
		dif = endTime - startTime;

		std::cout << "Vectorised Time:     " <<
			std::chrono::duration_cast<std::chrono::nanoseconds>(dif).count() <<
			"ns" << std::endl;

		std::cout << std::endl;

		if (subtractionNonVectorised != subtractionVectorised)
			std::cout << "Differences" << std::endl;
	}


	/**** Multiplications *****/
	{
		auto calcCount = ((unsigned long long) _m1->rows() * (unsigned long long) _m1->cols() * (unsigned long long) _m2->cols());
		auto origPrecision = std::cout.precision();
		std::cout.precision(3);
		std::cout << "Calc Count: " << (double) calcCount / 1000000000 << " GFLOPs" << std::endl;
		std::cout.precision(origPrecision);

		std::cout << "Performing matrix multiplication (Non-vectorised / naive loop)" << std::endl;
		auto startTime = std::chrono::high_resolution_clock::now();
		auto m3 = matrix::multiplyNonVectorised(_m1.get(), _m2.get());
		auto endTime = std::chrono::high_resolution_clock::now();
		auto dif = endTime - startTime;
		std::cout <<
			"Finished - " <<
			std::chrono::duration_cast<std::chrono::milliseconds>(dif).count() <<
			"ms (" <<
			((double)calcCount / std::chrono::duration_cast<std::chrono::nanoseconds>(dif).count()) <<
			"GFLOPs)" <<
			std::endl;

		std::cout << "Performing matrix multiplication (Non-vectorised / v2)" << std::endl;
		startTime = std::chrono::high_resolution_clock::now();
		auto m4 = matrix::multiplyNonVectorised2(_m1.get(), _m2.get());
		endTime = std::chrono::high_resolution_clock::now();
		dif = endTime - startTime;
		std::cout <<
			"Finished - " <<
			std::chrono::duration_cast<std::chrono::milliseconds>(dif).count() <<
			"ms (" <<
			((double)calcCount / std::chrono::duration_cast<std::chrono::nanoseconds>(dif).count()) <<
			"GFLOPs)" <<
			std::endl;

		std::cout << "Performing matrix multiplication (vectorised - naive loop)" << std::endl;
		startTime = std::chrono::high_resolution_clock::now();
		auto m5 = matrix::multipleVectorised256(_m1.get(), _m2.get());
		endTime = std::chrono::high_resolution_clock::now();
		dif = endTime - startTime;
		std::cout <<
			"Finished - " <<
			std::chrono::duration_cast<std::chrono::milliseconds>(dif).count() <<
			"ms (" <<
			((double)calcCount / std::chrono::duration_cast<std::chrono::nanoseconds>(dif).count()) <<
			"GFLOPs)" <<
			std::endl;

		std::cout << "Performing matrix multiplication (vectorised - v2)" << std::endl;
		startTime = std::chrono::high_resolution_clock::now();
		auto m6 = matrix::multipleVectorised256v2(_m1.get(), _m2.get());
		endTime = std::chrono::high_resolution_clock::now();
		dif = endTime - startTime;
		std::cout <<
			"Finished - " <<
			std::chrono::duration_cast<std::chrono::milliseconds>(dif).count() <<
			"ms (" <<
			((double)calcCount / std::chrono::duration_cast<std::chrono::nanoseconds>(dif).count()) <<
			"GFLOPs)" <<
			std::endl;

		std::cout << std::endl << "Checking Differences.." << std::endl;

		if (m3 != m4) {
			std::cout << "Differences between m3 and m4" << std::endl;
		}

		if (m3 != m5) {
			std::cout << "Differences between m3 and m5" << std::endl;
		}

		if (m3 != m6) {
			std::cout << "Differences between m3 and m6" << std::endl;
		}

		if (m4 != m5) {
			std::cout << "Differences between m4 and m5" << std::endl;
		}

		if (m4 != m6) {
			std::cout << "Differences between m4 and m6" << std::endl;
		}

		if (m5 != m6) {
			std::cout << "Differences between m5 and m6" << std::endl;
		}
	}

	return 0;
}

