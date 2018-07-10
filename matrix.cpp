#include "stdafx.h"

#include <malloc.h>
#include <intrin.h>
#include <iostream>
#include <exception>

#include "matrix.h"

#pragma region Constructors / Destructors

matrix::matrix(matrix && original) :
	_cols(original._cols),
	_rows(original._rows),
	_rowDataWidth(original._rowDataWidth),
	_data(original._data) {
	original._data = nullptr;
}

matrix::matrix(matrix & original) :
	_rows(original._rows),
	_cols(original._cols),
	_rowDataWidth(original._rowDataWidth),
	_data(nullptr) {
	_data = (float*)_aligned_malloc(_rows * _rowDataWidth * sizeof(float), 32);
	memcpy(_data, original._data, _rows * _rowDataWidth * sizeof(float));
}

matrix::matrix(unsigned int rows, unsigned int cols) :
	matrix(rows, cols, true) {
}

matrix::matrix(unsigned int rows, unsigned int cols, bool initialiseMemory) :
	_rows(rows),
	_cols(cols),
	_rowDataWidth(8 * ((cols + 7) / 8)) {
	_data = (float*)_aligned_malloc(rows * _rowDataWidth * sizeof(float), 32);

	if (initialiseMemory)
		memset(_data, 0, rows * _rowDataWidth * sizeof(float));
}

matrix::~matrix() {
	if (_data != nullptr) {
		_aligned_free((void*)_data);
		_data = nullptr;
	}
}

#pragma endregion

#pragma region Multiplication

matrix matrix::multiplyNonVectorised2(const matrix * matrix1, const matrix * matrix2) {
	if (matrix1 == nullptr)
		throw std::exception("matrix1 was null");

	if (matrix2 == nullptr)
		throw std::exception("matrix2 was null");

	if (matrix1->cols() != matrix2->rows())
		throw std::exception("Cannot multiple incompatible matrices");

	auto numCalcs = matrix1->cols();
	auto outputRowCount = matrix1->rows();
	auto outputColCount = matrix2->cols();
	matrix output(outputRowCount, outputColCount, false);

	auto m1Data = matrix1->getData();
	auto m2Data = matrix2->getData();
	auto outputData = output.getData();

	auto m1RowDataWidth = matrix1->rowDataWidth();
	auto m2RowDataWidth = matrix2->rowDataWidth();
	auto outputRowDataWidth = output.rowDataWidth();

	for (unsigned int outputRowIdx = 0; outputRowIdx < outputRowCount; ++outputRowIdx) {
		for (unsigned int idx = 0; idx < numCalcs; ++idx) {
			auto m1Value = m1Data[(outputRowIdx * m1RowDataWidth) + idx];
			for (unsigned int outputColIdx = 0; outputColIdx < outputColCount; ++outputColIdx) {
				auto val = idx == 0 ?
					0 :
					outputData[(outputRowIdx * outputRowDataWidth) + outputColIdx];

				auto m2Value = m2Data[(idx * m2RowDataWidth) + outputColIdx];
				val += m1Value * m2Value;

				outputData[(outputRowIdx * outputRowDataWidth) + outputColIdx] = val;
			}
		}
	}

	return output;
}

matrix matrix::multiplyNonVectorised(const matrix* matrix1, const matrix* matrix2) {
	if (matrix1 == nullptr)
		throw std::exception("matrix1 was null");

	if (matrix2 == nullptr)
		throw std::exception("matrix2 was null");

	if (matrix1->cols() != matrix2->rows())
		throw std::exception("Cannot multiple incompatible matrices");

	unsigned int numCalcs = matrix1->cols();

	unsigned int outputRowCount = matrix1->rows();
	unsigned int outputColCount = matrix2->cols();
	matrix output(outputRowCount, outputColCount, false);

	auto m1Data = matrix1->getData();
	auto m2Data = matrix2->getData();
	auto outputData = output.getData();

	auto m1RowDataWidth = matrix1->rowDataWidth();
	auto m2RowDataWidth = matrix2->rowDataWidth();
	auto outputRowDataWidth = output.rowDataWidth();

	for (unsigned int outputRowIdx = 0; outputRowIdx < outputRowCount; ++outputRowIdx) {
		for (unsigned int outputColIdx = 0; outputColIdx < outputColCount; ++outputColIdx) {

			float val(0);
			for (unsigned int idx = 0; idx < numCalcs; ++idx) {
				auto m1Value = m1Data[(outputRowIdx*m1RowDataWidth) + idx];
				auto m2Value = m2Data[(idx * m2RowDataWidth) + outputColIdx];
				val += m1Value * m2Value;
			}

			outputData[(outputRowIdx * outputRowDataWidth) + outputColIdx] = val;
		}
	}

	return output;
}

matrix matrix::multipleVectorised256(const matrix * matrix1, const matrix * matrix2) {
	if (matrix1 == nullptr)
		throw std::exception("matrix1 was null");

	if (matrix2 == nullptr)
		throw std::exception("matrix2 was null");

	if (matrix1->cols() != matrix2->rows())
		throw std::exception("Cannot multiple incompatible matrices");

	unsigned int numCalcs = matrix1->cols();

	unsigned int outputRowCount = matrix1->rows();
	unsigned int outputColCount = matrix2->cols();
	matrix output(outputRowCount, outputColCount);

	auto data1 = matrix1->getData();
	auto data2 = matrix2->getData();
	auto outputData = output.getData();

	auto m1RowDataWidth = matrix1->rowDataWidth();
	auto m2RowDataWidth = matrix2->rowDataWidth();
	auto outputRowDataWidth = output.rowDataWidth();

	// Loop over the output rows
	for (unsigned int outputRowIdx = 0; outputRowIdx < outputRowCount; ++outputRowIdx) {
		// Loop over the output columns (stepping using a size of 8 as that's the level of SIMD parallelisation)
		for (unsigned int outputColIdx = 0; outputColIdx < outputColCount; outputColIdx += 8) {
			auto output = _mm256_set1_ps(0);
			for (unsigned int joinIdx = 0; joinIdx < numCalcs; ++joinIdx) {
				auto m2Row = _mm256_load_ps(data2 + (joinIdx * m2RowDataWidth) + outputColIdx);
				auto m1Cell = _mm256_broadcast_ss(data1 + (outputRowIdx * m1RowDataWidth) + joinIdx);

				auto temp = _mm256_mul_ps(m1Cell, m2Row);
				output = _mm256_add_ps(output, temp);
			}

			_mm256_store_ps(outputData + (outputRowIdx * outputRowDataWidth) + outputColIdx, output);
		}
	}

	return output;
}

matrix matrix::multipleVectorised256v2(const matrix* matrix1, const matrix* matrix2) {
	if (matrix1 == nullptr)
		throw std::exception("matrix1 was null");

	if (matrix2 == nullptr)
		throw std::exception("matrix2 was null");

	if (matrix1->cols() != matrix2->rows())
		throw std::exception("Cannot multiple incompatible matrices");

	unsigned int numCalcs = matrix1->cols();

	unsigned int outputRowCount = matrix1->rows();
	unsigned int outputColCount = matrix2->cols();
	matrix output(outputRowCount, outputColCount, false);

	auto data1 = matrix1->getData();
	auto data2 = matrix2->getData();
	auto outputData = output.getData();

	auto m1RowDataWidth = matrix1->rowDataWidth();
	auto m2RowDataWidth = matrix2->rowDataWidth();
	auto outputRowDataWidth = output.rowDataWidth();

	// Loop over the output rows
	for (unsigned int outputRowIdx = 0; outputRowIdx < outputRowCount; ++outputRowIdx) {
		for (unsigned int joinIdx = 0; joinIdx < numCalcs; ++joinIdx) {
			// Loop over the output columns (stepping using a size of 8 as that's the level of SIMD parallelisation)

			auto m1Cell = _mm256_broadcast_ss(data1 + (outputRowIdx * m1RowDataWidth) + joinIdx);
			for (unsigned int outputColIdx = 0; outputColIdx < outputColCount; outputColIdx += 8) {

				auto val = joinIdx == 0 ?
					_mm256_set1_ps(0) :
					_mm256_load_ps(outputData + (outputRowIdx *outputRowDataWidth) + outputColIdx);

				auto m2Row = _mm256_load_ps(data2 + (joinIdx * m2RowDataWidth) + outputColIdx);

				auto additional = _mm256_mul_ps(m1Cell, m2Row);
				val = _mm256_add_ps(val, additional);

				_mm256_store_ps(outputData + (outputRowIdx * outputRowDataWidth) + outputColIdx, val);
			}
		}
	}

	return output;
}

#pragma endregion

#pragma region Addition Code

matrix matrix::addVectorised256(const matrix* matrix1, const matrix* matrix2) {
	if (matrix1 == nullptr)
		throw std::exception("matrix1 was null");

	if (matrix2 == nullptr)
		throw std::exception("matrix2 was null");

	if (matrix1->cols() != matrix2->cols())
		throw std::exception("Different number of columns");

	if (matrix1->rows() != matrix1->rows())
		throw std::exception("Different number of rows");

	matrix output(matrix1->rows(), matrix2->cols(), false);

	auto data1 = matrix1->getData();
	auto data2 = matrix2->getData();
	auto outputData = output.getData();

	auto dataLength = matrix1->rows() * matrix1->rowDataWidth();
	for (unsigned int offset = 0; offset < dataLength; offset += 8) {
		auto m1Cells = _mm256_load_ps(data1 + offset);
		auto m2Cells = _mm256_load_ps(data2 + offset);

		auto m3Cells = _mm256_add_ps(m1Cells, m2Cells);
		_mm256_store_ps(outputData + offset, m3Cells);
	}

	return output;
}

matrix matrix::addNonVectorised(const matrix* matrix1, const matrix* matrix2) {
	if (matrix1 == nullptr)
		throw std::exception("matrix1 was null");

	if (matrix2 == nullptr)
		throw std::exception("matrix2 was null");

	if (matrix1->cols() != matrix2->cols())
		throw std::exception("Different number of columns");

	if (matrix1->rows() != matrix2->rows())
		throw std::exception("Different number of rows");

	matrix output(matrix1->rows(), matrix2->cols(), false);

	auto data1 = matrix1->getData();
	auto data2 = matrix2->getData();
	auto outputData = output.getData();

	auto dataLength = matrix1->rows() * matrix1->rowDataWidth();
	for (unsigned int offset = 0; offset < dataLength; ++offset) {
		outputData[offset] = data1[offset] + data2[offset];
	}

	return output;
}

#pragma endregion

# pragma region Subtraction Code

matrix matrix::subtractVectorised256(const matrix* matrix1, const matrix* matrix2) {
	if (matrix1 == nullptr)
		throw std::exception("matrix1 was null");

	if (matrix2 == nullptr)
		throw std::exception("matrix2 was null");

	if (matrix1->cols() != matrix2->cols())
		throw std::exception("Different number of columns");

	if (matrix1->rows() != matrix1->rows())
		throw std::exception("Different number of rows");

	matrix output(matrix1->rows(), matrix2->cols());

	auto data1 = matrix1->getData();
	auto data2 = matrix2->getData();
	auto outputData = output.getData();

	auto dataLength = matrix1->rows() * matrix1->rowDataWidth();
	for (unsigned int offset = 0; offset < dataLength; offset += 8) {
		auto m1Cells = _mm256_load_ps(data1 + offset);
		auto m2Cells = _mm256_load_ps(data2 + offset);

		auto m3Cells = _mm256_sub_ps(m1Cells, m2Cells);
		_mm256_store_ps(outputData + offset, m3Cells);
	}

	return output;
}

matrix matrix::subtractNonVectorised(const matrix* matrix1, const matrix* matrix2) {
	if (matrix1 == nullptr)
		throw std::exception("matrix1 was null");

	if (matrix2 == nullptr)
		throw std::exception("matrix2 was null");

	if (matrix1->cols() != matrix2->cols())
		throw std::exception("Different number of columns");

	if (matrix1->rows() != matrix1->rows())
		throw std::exception("Different number of rows");

	matrix output(matrix1->rows(), matrix2->cols());

	auto data1 = matrix1->getData();
	auto data2 = matrix2->getData();
	auto outputData = output.getData();

	auto dataLength = matrix1->rows() * matrix1->rowDataWidth();
	for (unsigned int offset = 0; offset < dataLength; ++offset) {
		outputData[offset] = data1[offset] - data2[offset];
	}

	return output;
}

# pragma endregion

#pragma region Operators

// Strictly speaking, one would like to have the code be selected
// based off the supported capabilities of the CPU, i.e. if AVX2 
// was supposed for a given runtime, then that should be used, otherwise
// AVX etc.

matrix operator*(const matrix& matrix1, const matrix& matrix2)
{
	if (matrix1.cols() != matrix2.rows())
		throw std::exception("Cannot multiple incompatible matrices");

	return matrix::multipleVectorised256v2(&matrix1, &matrix2);
}

matrix operator+(const matrix& matrix1, const matrix& matrix2)
{
	if (matrix1.cols() != matrix2.cols() ||
		matrix1.rows() != matrix2.rows())
		throw std::exception("Cannot add incompatible matrices");

	return matrix::addVectorised256(&matrix1, &matrix2);
}

matrix operator-(const matrix& matrix1, const matrix& matrix2)
{
	if (matrix1.cols() != matrix2.cols() ||
		matrix1.rows() != matrix2.rows())
		throw std::exception("Cannot subtract incompatible matrices");

	return matrix::subtractVectorised256(&matrix1, &matrix2);
}

bool operator==(const matrix & matrix1, const matrix& matrix2)
{
	if (matrix1.rows() != matrix2.rows() ||
		matrix1.cols() != matrix2.cols()) {
		std::cout << "Different size" << std::endl;
		return false;
	}

	// For the comparison, although it would appear to be sensible to be able to
	// simply compare each piece of memory, because we need to align each row
	// of the matrix in memory (for SIMD loading) then we need to ensure that
	// we don't compare the buffer values as they're undefined and so will lead
	// to false failures
	int rowCount = matrix1.rows();
	int colCount = matrix1.cols();
	int m1RowDataWidth = matrix1.rowDataWidth();
	int m2RowDataWidth = matrix2.rowDataWidth();

	int m1Offset(0), m2Offset(0);

	int m1RowPadding = m1RowDataWidth - matrix1.cols();
	int m2RowPadding = m2RowDataWidth - matrix2.cols();

	float* data1 = matrix1.getData();
	float* data2 = matrix2.getData();

	for (int rowIdx = 0; rowIdx < rowCount; ++rowIdx) {
		for (int colIdx = 0; colIdx < colCount; ++colIdx) {
			if (data1[m1Offset++] != data2[m2Offset++])
				return false;
		}

		m1Offset += m1RowPadding;
		m2Offset += m2RowPadding;
	}

	return true;
}

bool operator!=(const matrix & matrix1, const matrix& matrix2)
{
	return !(matrix1 == matrix2);
}

#pragma endregion