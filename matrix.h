#pragma once

class matrix {
private:
	unsigned int _rows, _cols, _rowDataWidth;

	float* _data;

public:
	matrix(matrix&& original);
	matrix(matrix& original);
	matrix(unsigned int rows, unsigned int cols);
	matrix(unsigned int rows, unsigned int cols, bool resetMemory);
	~matrix();

	unsigned int rows() const { return this->_rows; }
	unsigned int cols() const { return this->_cols; }
	unsigned int rowDataWidth() const { return this->_rowDataWidth; }

	float value(unsigned int row, unsigned int col) const { return _data[(row * _rowDataWidth) + col]; }
	void setValue(unsigned int row, unsigned int col, float val) { _data[(row * _rowDataWidth) + col] = val; }

	float* getData() const { return _data; };

	static matrix multiplyNonVectorised(const matrix* matrix1, const matrix* matrix2);
	static matrix multiplyNonVectorised2(const matrix* matrix1, const matrix* matrix2);
	static matrix multipleVectorised256(const matrix* matrix1, const matrix* matrix2);
	static matrix multipleVectorised256v2(const matrix* matrix1, const matrix* matrix2);

	static matrix addNonVectorised(const matrix* matrix1, const matrix* matrix2);
	static matrix addVectorised256(const matrix* matrix1, const matrix* matrix2);

	static matrix subtractNonVectorised(const matrix* matrix1, const matrix* matrix2);
	static matrix subtractVectorised256(const matrix* matrix1, const matrix* matrix2);
};

// Operator overloads

matrix operator+(const matrix& matrix1, const matrix& matrix2);
matrix operator-(const matrix& matrix1, const matrix& matrix2);
matrix operator*(const matrix& matrix1, const matrix& matrix2);
bool operator==(const matrix& matrix1, const matrix& matrix2);
bool operator!=(const matrix& matrix1, const matrix& matrix2);