#include <vector>
#include <cmath>
#include <random>
#include <fstream>
using namespace std;

enum class PivotingType
{
	None,
	Partial,
	ScaledPartial,
	Full
};

template <class type = double>
class Matrix
{
public:
	vector< vector<type> > Elements;
private:
	int Height;
    int Width;
	bool oddSwap;
	
public:

	#pragma region Constructor/Destructor

	//Blank constructor
	Matrix()
	{
		oddSwap = false;
		Resize(0,0);
	}

	//Sets matrix dimensions
	Matrix(int height,int width)
    {
        oddSwap = false;
		Resize(height,width);
	}

	//Constructor with file name to read
	Matrix(char *filename)
	{
		oddSwap = false;
		ReadFromFile(filename);
	}

	//Destruct
	~Matrix(){}

	#pragma endregion

	#pragma region Getter/setter

	//Get private height
    int GetHeight() const 
	{
		return Height;
	}

	//Get private width
	int GetWidth() const
	{
		return Width;
	}

	//Resize matrix
	void Resize(int newHeight, int newWidth)
	{
		Height = newHeight;
		Width = newWidth;
		Elements.resize(Height);
		for (int i = 0; i < Height; i++)
		{
			Elements[i].resize(Width);
		}
	}

	#pragma endregion

	#pragma region Operations

	// Returns transposed matrix
	Matrix Transpose()
	{
		Matrix transposed(Width,Height);

		for (int i = 0; i < transposed.Height; i++)
		{
			for (int j = 0; j < transposed.Width; j++)
			{
				transposed.Elements[i][j] = Elements[j][i];
			}
		}

		return transposed;
	}

	// Check if matrix elements are equal with epsion difference
	bool Equals(const Matrix& b,type epsilon = 0)
	{
		if(Height != b.Height || Width != b.Width)
		{
			return false;
		}
			
		for (int i = 0; i < Height; i++)
		{
			for (int j = 0; j < Width; j++)
			{
				if(fabs(Elements[i][j] - b.Elements[i][j]) > epsilon)
				{
					return false;
				}
			}
		}

		return true;
	}

	// Append two matrixes into first one
	void AugmentWith(const Matrix& b)
	{
		if(Height != b.Height)
		{
			throw exception("Matrixes could not be appended. Error in dimensions");
		}

		int oldWidth = Width;

		Resize(Height, oldWidth + b.Width);

		for (int i = 0; i < Height; i++)
		{
			for (int j = oldWidth; j < Width; j++)
			{
				Elements[i][j] = b.Elements[i][j - oldWidth];
			}
		}
	}

	// Get rank of the matrix TODO: Should be revised
	int GetRank(type epsilon = 0)
	{
		int rankCounter = 0;
		Matrix<type> eliminatedMatrix = DecompositeLUP(PivotingType::Full).ExtractUpper();

		for (int i = 0; i < eliminatedMatrix.Height; i++)
		{
			for (int j = i; j < eliminatedMatrix.Width; j++)
			{
				if (fabs(eliminatedMatrix.Elements[i][j] - 0) > epsilon)
				{
					rankCounter++;
					break;
				}
			}
		}

		return rankCounter;
	}

	// Extract matrix row as vector
	vector<type> ExtractRow(int extractIndex)
	{
		if(extractIndex < 0 || extractIndex >= Height)
		{
			throw exception("Row can not be extracted. Index out of bounds");			
		}
		
		return Elements[extractIndex];
	}

	// Extract matrix column as vector
	vector<type> ExtractColumn(int extractIndex)
	{
		if(extractIndex < 0 || extractIndex >= Width)
		{
			throw exception("Column can not be extracted. Index out of bounds");
		}

		vector<type> column(Height);
		for (int i = 0; i < Height; i++)
		{
			column[i] = Elements[i][extractIndex];
		}

		return column;		
	}

	// Extract diagonl matrix, while other elements are 0
	Matrix ExtractDiagonal(type factor = 1)
	{
		if(Height != Width)
		{
			throw exception("Coul not extract diagonal. Matrix is not square");
		}

		Matrix diag(Height, Width);
		for (int i = 0; i < Height; i++)
		{
			diag.Elements[i][i] = Elements[i][i] * factor;
		}

		return diag;
	}

	// Extract lower part of Matrix
	Matrix ExtractLower(type factor = 1)
	{
		Matrix lower = Diagonal(Height, Width);

		for (int i = 1; i < Height; i++)
		{
			for (int j = 0; j < fmin(i, Width); j++)
			{
				lower.Elements[i][j] = Elements[i][j] * factor;
			}
		}

		return lower;
	}

	// Extract upper part of Matrix
	Matrix ExtractUpper(type factor = 1)
	{
		Matrix upper(Height, Width);

		for (int i = 0; i < fmin(Height, Width); i++)
		{
			for (int j = i; j < Width; j++)
			{
				upper.Elements[i][j] = Elements[i][j] * factor;
			}
		}

		return upper;
	}

	// Extract matrix with indexes
	Matrix ExtractMatrix(int startingIndexI, int startingIndexJ, int height, int width)
	{
		if (startingIndexI >= Height || startingIndexJ >= Width)
		{
			throw exception("Matrix can not be extracted. Error with index bounds.");
		}

		Matrix extract(height, width);
		for (int i = 0; i < fmin(Height - startingIndexI, extract.Height); i++)
		{
			for (int j = 0; j < fmin(Width - startingIndexJ, extract.Width); j++)
			{
				extract[i][j] = Elements[startingIndexI + i][startingIndexJ + j];
			}
		}

		return extract;
	}

	// Insert matrix with indexes
	void InsertMatrix(const Matrix& insert, int i1, int j1)
	{
		if(i1 + insert.Height > Height || j1 + insert.Width > Width)
		{
			throw exception("Matrix can not be inserted. Error with index bounds.");
		}

		int elementsCount = 0;
		for (int i = i1; i < i1 + insert.Height; i++)
		{
			for (int j = j1; j < j1 + insert.Width; j++)
			{
				Elements[i][j] = insert.Elements[elementsCount / insert.Width][elementsCount % insert.Width];
				elementsCount++;
			}
		}
	}

	// Swap row a and row b 
	void SwapRows(int a, int b)
	{
		if(a < 0 || a >= Height || b < 0 || b >= Height)
		{
			throw exception("Could not permute rows. Indexes out of bound");
		}

		if (a != b)
		{
			for (int j = 0; j< Width; j++)
			{
				type temp = Elements[a][j];
				Elements[a][j] = Elements[b][j];
				Elements[b][j] = temp;
			}
			oddSwap = !oddSwap; //If number of swaps is odd then determinant is reversed
		}
	}

	// Swap column a and column b 
	void SwapColumns(int a, int b)
	{
		if(a < 0 || a >= Width || b < 0 || b >= Width)
		{
			throw exception("Could not permute columns. Indexes out of bound");
		}

		if (a != b)
		{
			for (int i = 0; i < Height; i++)
			{
				type temp = Elements[i][a];
				Elements[i][a] = Elements[i][b];
				Elements[i][b] = temp;
			}
			oddSwap = !oddSwap; //If number of swaps is odd then determinant is reversed
		}
	}

	// Matrix trace
	type Trace()
	{
		if(Height != Width)
		{
			throw exception("Only square matrix has trace");
		}

		type trace = 0;
		for (int i = 0; i < Height; i++)
		{
			trace += Elements[i][i];
		}

		return trace;
	}

	// Vector scalar multiplication
	type VectorMultiplyScalar(const Matrix& b)
	{
		type scalar = 0;
		if(Height == 1 && b.Height == 1 && Width == b.Width)
		{
			for (int i = 0; i < Width; i++)
			{
				scalar += Elements[0][i] * b.Elements[0][i];
			}
		}
		else if(Width == 1 && b.Height == 1 && Height == b.Width)
		{
			for (int i = 0; i < Height; i++)
			{
				scalar += Elements[i][0] * b.Elements[0][i];
			}
		}
		else if(Height == 1 && b.Width == 1 && Width == b.Height)
		{
			for (int i = 0; i < Width; i++)
			{
				scalar += Elements[0][i] * b.Elements[i][0];
			}
		}
		else if(Width == 1 && b.Width == 1 && Height == b.Height)
		{
			for (int i = 0; i < Height; i++)
			{
				scalar += Elements[i][0] * b.Elements[i][0];
			}
		}
		else
		{
			throw exception("Scalar multiplication could not be done");
		}
		return scalar;
	}

	// Returns matrix multyplied by other matrix using general method with unning time O(n^3)
	Matrix MultiplyBy(Matrix& b)
	{
		if (Width != b.Height) //if width of the first matrix equals height of the second
		{
			throw exception("Matrixes can not by multiplied. Width of the first is not the same as seconds height");
		}

		Matrix retMultiplied(Height, b.Width);

		for (int i = 0; i < retMultiplied.Height; i++)
		{
			for (int j = 0; j < retMultiplied.Width; j++)
			{
				type sum = 0;
				for (int k = 0; k < Width; k++)
				{
					sum += Elements[i][k] * b.Elements[k][j];
				}
				retMultiplied.Elements[i][j] = sum;
			}
		}

		return retMultiplied;
	}

	// Returns matrix multyplied by other matrix using Strassen method with unning time O(n^Log7)
	Matrix MultiplyByStrassen(Matrix& b, int baseCaseDepth = 1)
	{
		if (Height != Width || Width != b.Height || Height != b.Width) //if width of the first matrix equals height of the second
		{
			throw exception("Matrixes can not by multiplied using Strassen algorithm!");
		}

		Matrix retMultiplied(Height, b.Width);

		// TODO: Review for non power of two dimensions, speed up
		if (Height <= baseCaseDepth)
		{
			return MultiplyBy(b);
		}

		//Extract submatrixes using zero pading
		int segmentHeight = Height / 2 + Height % 2;
		Matrix A = ExtractMatrix(0, 0, segmentHeight, segmentHeight);
		Matrix B = ExtractMatrix(0, segmentHeight, segmentHeight, segmentHeight);
		Matrix C = ExtractMatrix(segmentHeight, 0, segmentHeight, segmentHeight);
		Matrix D = ExtractMatrix(segmentHeight, segmentHeight, segmentHeight, segmentHeight);

		Matrix E = b.ExtractMatrix(0, 0, segmentHeight, segmentHeight);
		Matrix F = b.ExtractMatrix(0, segmentHeight, segmentHeight, segmentHeight);
		Matrix G = b.ExtractMatrix(segmentHeight, 0, segmentHeight, segmentHeight);
		Matrix H = b.ExtractMatrix(segmentHeight, segmentHeight, segmentHeight, segmentHeight);

		Matrix M1 = A.MultiplyByStrassen(F - H);
		Matrix M2 = (A + B).MultiplyByStrassen(H);
		Matrix M3 = (C + D).MultiplyByStrassen(E);
		Matrix M4 = D.MultiplyByStrassen(G - E);
		Matrix M5 = (A + D).MultiplyByStrassen(E + H);
		Matrix M6 = (B - D).MultiplyByStrassen(G + H);
		Matrix M7 = (A - C).MultiplyByStrassen(E + F);

		retMultiplied.InsertMatrix(M5 + M4 - M2 + M6, 0, 0);
		retMultiplied.InsertMatrix(M1 + M2, 0, retMultiplied.Width / 2);
		retMultiplied.InsertMatrix(M3 + M4, retMultiplied.Height / 2, 0);
		retMultiplied.InsertMatrix(M1 + M5 - M3 - M7, retMultiplied.Height / 2, retMultiplied.Width / 2);

		return retMultiplied;
	}

	// Returns matrix multiplied by number
	Matrix MultiplyBy(type number)
	{
		Matrix retMultiplied(Height, Width);
		for (int i = 0; i < Height; i++)
		{
			for (int j = 0; j < Width; j++)
			{
				retMultiplied.Elements[i][j] = Elements[i][j] * number;
			}
		}

		return retMultiplied;
	}

	// Get matrix determinant
	type Determinant()
	{
		if(Height != Width)
		{
			throw exception("Determinant can not be computed. Matrix is not square");
		}

		try
		{
			type det = 1;
			Matrix LU = DecompositeLUP(PivotingType::Full);
			for (int i = 0; i < Height; i++) //if non singular multiply U[i][i];
			{
				det *= LU.Elements[i][i];
			}
			
			//If amount of swaps made are odd then reverse determinant
			return oddSwap ? -det : det;
		}
		catch (exception e) //singular matrix case
		{
			return 0;
		}
	}

	// Get minor matrix at i,j
	Matrix Minor(int iTh, int jTh)
	{
		if(iTh < 0 || iTh >= Height || jTh < 0 || jTh >= Width)
		{
			throw exception("Could not get minor. Indexes are out of range");
		}

		Matrix minor(Height - 1, Width - 1);
		int count = 0;
		for (int i = 0; i < Height; i++)
		{
			for (int j = 0; j < Width; j++)
			{
				if (i != iTh && j != jTh)
				{
					minor.Elements[count / minor.Width][count % minor.Height] = Elements[i][j];
					count++;
				}
			}
		}

		return minor;
	}

	// Find inverse of the matrix with LU solving
	Matrix Inverse()
	{
		return SolveDirectGauss(Diagonal(Height, Width), DecompositeLUP());
	}

	// Get matrixes conditional number
	type ConditionNumber()
	{
		return NormInfinity() * Inverse().NormInfinity();
	}

	// Raise matrix to k-th power
	Matrix Power(int k)
	{
		if(Height != Width)
		{
			throw exception("Matrix can not be raised to the power. Matrix is not square");
		}

		Matrix result = Diagonal(Height, Width);
		Matrix unit = *this;

		if (k < 0)
		{
			unit = Inverse();
			k = -k;
		}

		while (k > 0)
		{
			// if k is odd
			if (k & 1)
			{
				result = result.MultiplyBy(unit);
			}

			if(!(k >>= 1)) continue;
			unit = unit.MultiplyBy(unit);
		}

		return result;
	}

	// Find eigenvalues using power iteration
	type PowerMethod(type epsilon, int maxIteration = INT_MAX)
	{
		if(Height != Width)
		{
			throw exception("Power method could not be done on non square matrix");
		}

		throw exception("Method not implemented exception"); //TODO
	}

	// Calculate final state from transition matrix for Markov chain
	Matrix MarkovStationaryMatrix()
	{
		if(IsLeftStochastic())
		{
			Matrix transition = *this;
			for (int i = 0; i < Height; i++)
			{
				transition.Elements[i][i] -= 1;
				transition.Elements[Height-1][i] = 1;
			}
			Matrix B(Height,1);
			B.Elements[Height-1][0] = 1;
			return transition.SolveDirectGauss(B);
		}
		
		if(IsRightStochastic())
		{
			Matrix transition = Transpose();
			for (int i = 0; i < Height; i++)
			{
				transition.Elements[i][i] -= 1;
				transition.Elements[Height-1][i] = 1;
			}
			Matrix B(Height,1);
			B.Elements[Height-1][0] = 1;
			return transition.SolveDirectGauss(B).Transpose();
		}
		
		throw exception("Stationary matrix can not be calculated. Matrix is not stochastic matrix");
	}

	// Calculate limiting matrix for Markov chain
	Matrix MarkovLimitingMatrix()
	{
		vector<int> absorbingIndex;
		bool isRightstochastic;

		if(IsAbsorbing(absorbingIndex, isRightstochastic))
		{
			int absorbingCount = absorbingIndex.size();
			int nonAbsorbingCount = Height - absorbingCount;

			//Absorbing states to the top left corner of matrix
			for (int i = 0; i < absorbingCount; i++) 
			{
				SwapRows(i, absorbingIndex[i]);
				SwapColumns(i, absorbingIndex[i]);
			}

			Matrix limiting(Height, Width);
			limiting.InsertMatrix(Diagonal(absorbingCount), 0,0);
			//F = (I - Q)^-1
			Matrix f = (Diagonal(nonAbsorbingCount) - ExtractMatrix(absorbingCount, absorbingCount, nonAbsorbingCount - 1, nonAbsorbingCount - 1)).Inverse(); 
			if(isRightstochastic)
			{
				Matrix r = f.MultiplyBy(ExtractMatrix(absorbingCount, 0, nonAbsorbingCount - 1, absorbingCount - 1)); //F*R
				limiting.InsertMatrix(r, absorbingCount, 0);
			}
			else
			{
				Matrix r = ExtractMatrix(0, absorbingCount, absorbingCount - 1, nonAbsorbingCount - 1).MultiplyBy(f); //R*F
				limiting.InsertMatrix(r, 0, absorbingCount);
			}

			return limiting;
		}
		
		throw exception("Limiting matrix could not be found. Matrix is not absorbing");
	}

	// Output hypothesis koeficients using normal equation
	Matrix LearnFromData(Matrix functionOutput)
	{
		Matrix transpose = Transpose();
		
		return transpose.MultiplyBy(*this).SolveDirectGauss(transpose.MultiplyBy(functionOutput));
	}
	
	#pragma endregion

	#pragma region Decompositions

	// Matrix LU decomposition with pivoting
	Matrix DecompositeLUP(PivotingType pivotingType = PivotingType::None)
	{
		Matrix luComposite = Diagonal(Height, Width);

		//copy
		luComposite.Elements = Elements;

		//compute
		for (int j = 0; j < fmin(Height, Width) - 1; j++)
		{
			if (pivotingType == PivotingType::Partial)
			{
				//Pivoting row swap
				int maxIndex = j;

				for(int i = j + 1; i < Height; i++) //find maximal element index in column
				{
					if(fabs(luComposite.Elements[i][j]) > fabs(luComposite.Elements[maxIndex][j]))
						maxIndex = i;
				}

				if(maxIndex != j) //swap with maximal element in column if exists
				{
					luComposite.SwapRows(j,maxIndex);  //Swap elements of LU matrix
				} 
			}
			else if(pivotingType == PivotingType::Full)
			{
				//Pivoting row and column swap
				int maxRowIndex = j;
				int maxColumnIndex = j;

				for(int i = j; i < Height; i++) //find maximal element index in block matrix
				{
					for (int j2 = j; j2 < Width; j2++)
					{
						if(fabs(luComposite.Elements[i][j2]) > fabs(luComposite.Elements[maxRowIndex][maxColumnIndex]))
						{
							maxRowIndex = i;
							maxColumnIndex = j2;
						}
					}					
				}

				if(maxRowIndex != j) //swap with maximal element in column if exists
				{
					luComposite.SwapRows(j,maxRowIndex);  //Swap elements of LU matrix
				} 

				if(maxColumnIndex != j) //swap with maximal element in row if exists
				{
					luComposite.SwapColumns(j,maxColumnIndex);  //Swap elements of LU matrix
				}
			}

			//Pivot element now is max in it's column
			type pivot = luComposite.Elements[j][j];

			//If pivot is still zero then throw exception
			if(pivot == 0)
			{
				throw exception("LU Decomsposition error. Zero pivot");
			}

			//If pivot is non zero calculate LU
			for (int i = j+1; i < Height; i++)
			{
				type l = luComposite.Elements[i][j] / pivot;

				for (int k = j + 1; k < Width; k++)
				{
					luComposite.Elements[i][k] -= luComposite.Elements[j][k] * l; //Upper part
				}

				luComposite.Elements[i][j] = l; //Lower part
			}
		}

		return luComposite;		 
	}

	// Matrix LU decomposition with partial pivoting for linear systems
	Matrix DecompositeLUP(Matrix& permutation)
	{
		if (Height != Width || permutation.Height != Height)
		{
			throw exception("Could not solve linear epuation. Problem with dimensions");
		}

		Matrix luComposite(Height, Width);
		//copy
		luComposite.Elements = Elements;
		//compute
		for (int j = 0; j < Width - 1; j++)
		{
			//Pivoting row and column swap
			int maxRowIndex = j;

			for (int i = j; i < Height; i++) //find maximal element index in block matrix
			{
				if (fabs(luComposite.Elements[i][j]) > fabs(luComposite.Elements[maxRowIndex][j]))
				{
					maxRowIndex = i;
				}
			}
			if (maxRowIndex != j) //swap with maximal element in column if exists
			{
				luComposite.SwapRows(j, maxRowIndex);  //Swap elements of LU matrix
				permutation.SwapRows(j, maxRowIndex);  //Swap elements of permutation matrix
			}

			//Pivot element now is max in it's column
			type pivot = luComposite.Elements[j][j];

			//If pivot is still zero throw exception
			if (pivot == 0)
			{
				throw exception("LU Decomsposition error. Zero pivot");
			}

			//If pivot is non zero calculate LU
			for (int i = j + 1; i < Height; i++)
			{
				type l = luComposite.Elements[i][j] / pivot;
				for (int k = j + 1; k < Width; k++)
				{
					luComposite.Elements[i][k] -= luComposite.Elements[j][k] * l; //Upper part
				}
				luComposite.Elements[i][j] = l; //Lower part
			}
		}

		return luComposite;
	}

	// Matrix Cholesky decomposition
	Matrix DecompositeCholesky()
	{
		Matrix choleskyComposite(Height, Width);
		for (int j = 0; j < Width; j++)
		{
			type sum = 0;
			for (int k = 0; k < j; k++)
			{
				sum += pow(choleskyComposite.Elements[k][j],2);
			}

			type pivot = Elements[j][j] - sum;

			// If pivot is less then zero then square root can not be computed. Exception is beeing thrown
			if(pivot <= 0)
			{
				throw exception("Choleski decomposition error. Matrix is not positive definite");
			}

			choleskyComposite.Elements[j][j] = pivot = sqrt(pivot);

			for (int i = j + 1; i < Height; i++)
			{
				sum = 0;
				for (int k = 0; k < j; k++)
				{
					sum += choleskyComposite.Elements[i][k] * choleskyComposite.Elements[j][k];
				}

				choleskyComposite.Elements[i][j] = choleskyComposite.Elements[j][i] = (Elements[i][j] - sum) / pivot;
			}
		}

		return choleskyComposite;
	}

	// Matrix QR decomposition
	void DecompositeQR(Matrix& orthogonalQ, Matrix& upperR)
	{
		Matrix<type> Q(Height, Width);
		Matrix<type> R(Width, Width);



		orthogonalQ = Q;
		upperR = R;
	}

	#pragma endregion

	#pragma region Linear Solvers

	//Returns solution vector for linear equation for matrix
	Matrix SolveDirectGauss(Matrix& b)
	{
		if(Height != Width || Height != b.Height)
		{
			throw exception("Could not solve linear system with gauss method. Error with matrix dimensions");
		}
	
		Matrix x(Height, b.Width); //Height of A, Width of b
		vector<type> y(Height); //y = Ux;
		Matrix LU = DecompositeLUP(b);

		if (LU.Elements[Height - 1][Width - 1] == 0)
		{
			throw exception("Linear system can not be solved. Matrix is sungular");
		}

		for (int k = 0; k < b.Width; k++)
		{
			//y Forward substitution Part
			for (int i = 0; i < Height; i++)
			{
				type sum = 0;
				for (int j = 0; j < i; j++)
				{
					sum += LU.Elements[i][j] * y[j];
				}
				y[i] = b.Elements[i][k] - sum;
			}

			//x Backward substitution part
			for (int i = Height - 1; i >= 0; i--)
			{
				type sum = 0;
				for (int j = i + 1; j < Height; j++)
				{
					sum += LU.Elements[i][j] * x.Elements[j][k];
				}
				x.Elements[i][k] = (y[i] - sum) / LU.Elements[i][i];
			}
		}

		return x;
	}

	//Returns solution vector for linear equation with custom composite
	Matrix SolveDirectGauss(const Matrix& b, const Matrix& decomposite)
	{
		if(Height != Width || Height != b.Height || decomposite.Height != Height || decomposite.Width != Width)
		{
			throw exception("Could not solve linear system with gauss method. Error with A, b or composite matrix dimensions");
		}

		if (decomposite.Elements[Height - 1][Width - 1] == 0)
		{
			throw exception("Linear system can not be solved. Matrix is sungular");
		}

		Matrix x(Height, b.Width); //Height of A, Width of b
		vector<type> y(Height); //y = Ux;
		for (int k = 0; k < b.Width; k++)//if b width > 1
		{
			//y Forward substitution Part
			for (int i = 0; i < Height; i++)
			{
				type sum = 0;
				for (int j = 0; j < i; j++)
				{
					sum += decomposite.Elements[i][j] * y[j];
				}
				y[i] = b.Elements[i][k] - sum;
			}

			//x Backward substitution part
			for (int i = Height - 1; i >= 0; i--)
			{
				type sum = 0;
				for (int j = i + 1; j < Height; j++)
				{
					sum += decomposite.Elements[i][j] * x.Elements[j][k];
				}
				x.Elements[i][k] = (y[i] - sum) / decomposite.Elements[i][i];
			}
		}

		return x;
	}

	//Solve tridiagonal matrixes with thomas algorithm for tridiagonal matrixes
	Matrix SolveDirectThomas(const Matrix& b)
	{
		if(!IsDiagonal(3) || Height != Width || b.Height != Height)
		{
			throw exception("Could not solve linear system with Thomas method. Error with A or b matrix dimensions");
		}

		Matrix x(Height, b.Width); //Height of A, Width of b
		vector<type> y(Height); //y = Ux;
		Matrix luComposite(Height, Width);
		
		luComposite.Elements = Elements; //Elements are copied
		for (int i = 1; i < Height; i++) //Generating fast version of LU for tridiagonal
		{
			type pivot = luComposite.Elements[i - 1][i - 1];

			if (pivot == 0) //Stop is pivot is zero
			{
				throw exception("Error in Thomas method - pivot is zero");
			}

			type l = Elements[i][i - 1] / pivot;
			luComposite.Elements[i][i] = Elements[i][i] - l * Elements[i - 1][i];
			luComposite.Elements[i][i - 1] = l;
		}

		//if b width > 1
		for (int k = 0; k < b.Width; k++)
		{
			//y Forward substitution Part
			y[0] = b.Elements[0][k];
			for (int i = 1; i < Height; i++)
			{
				y[i] = b.Elements[i][k] - luComposite.Elements[i][i - 1] * y[i - 1];
			}

			//x Backward substitution part
			x.Elements[Height - 1][k] = y[Height - 1] / luComposite.Elements[Height - 1][Height - 1];
			for (int i = Height - 2; i >= 0; i--)
			{
				x.Elements[i][k] = (y[i] - luComposite.Elements[i][i + 1] * x.Elements[i + 1][k]) / luComposite.Elements[i][i];
			}
		}

		return x;
	}

	//Solves linear systems iterativly by Jacobi method with relaxation
	Matrix SolveIterativeJacobi(const Matrix& b, type epsilon, type relaxation = 1, int maxIteration = INT_MAX)
	{
		if(!IsStrictlyDominant() || Height != Width || b.Height != Height)
		{
			throw exception("Iterative Jacob method could not be done. A matrix does not converge");
		}

		Matrix x(Height, b.Width); //first zero matrix of the iteration
		vector<type> xNext(Height); //x(k + 1) vector;
		int iterationCount = 0;

		//iterate while error is more then epsilon and count less then maxIterations
		while ((*this * x - b).NormInfinity() > epsilon && iterationCount < maxIteration)
		{
			for (int k = 0; k < x.Width; k++)
			{
				for (int i = 0; i < x.Height; i++)
				{
					type sum = 0;
					for (int j = 0; j < Width; j++)
					{
						if (j != i)
						{
							sum += Elements[i][j] * x.Elements[j][k];
						}
					}

					xNext[i] = (b.Elements[i][k] - sum)*relaxation / Elements[i][i] + ((1 - relaxation)*x.Elements[i][k]);
				}

				//x(k) = x(k+1) for the Next iteration
				for (int i = 0; i < Height; i++)
				{
					x.Elements[i][k] = xNext[i];
				}
			}

			iterationCount++;
		}

		return x;
	}

	//Solves linear systems iterativly by Successive Over-Relaxation method (Gauss-Seidel if relaxation = 1)
	Matrix SolveIterativeSOR(const Matrix& b, type epsilon, type relaxation = 1, int maxIteration = INT_MAX)
	{
		if(!IsStrictlyDominant() || Height != Width || b.Height != Height)
		{
			throw exception("Iterative SOR method could not be done. A matrix does not converge");
		}

		Matrix x(Height, b.Width); //first zero matrix of the iteration
		vector<type> xNext(Height); //x(k + 1) vector
		int iterationCount = 0;

		//iterate while error is more then epsilon and count less then max
		while ((*this * x - b).NormInfinity() > epsilon && iterationCount < maxIteration)
		{
			for (int k = 0; k < x.Width; k++)
			{
				for (int i = 0; i < x.Height; i++)
				{
					type sumNext = 0;
					for (int j = 0; j < i; j++)
					{
						sumNext += Elements[i][j] * xNext[j];
					}
					type sum = 0;
					for (int j = i + 1; j < Width; j++)
					{
						sum += Elements[i][j] * x.Elements[j][k];
					}

					xNext[i] = (b.Elements[i][k] - sumNext - sum)*relaxation / Elements[i][i] + ((1 - relaxation)*x.Elements[i][k]);
				}

				//x(k) = x(k+1) for the Next iteration
				for (int i = 0; i < Height; i++)
				{
					x.Elements[i][k] = xNext[i];
				}
			}

			iterationCount++;
		}

		return x;
	}

	#pragma endregion
		
	#pragma region Norms

	//Vector entrywise p-th norm
	type Norm(double p = 2)
	{
		if(p < 1)
		{
			throw exception("Entrywise p-th norm could not be calculated. P < 1");
		}

		type norm = 0;
		for (int i = 0; i < Height; i++)
		{
			for (int j = 0; j < Width; j++)
			{
				norm += pow(fabs(Elements[i][j]), p);
			}
		}

		return pow(norm, 1 / p);
	}

	//Matrix first norm
	type Norm1()
	{
		type max = 0;
		for (int j = 0; j < Width; j++)
		{
			type sum = 0;
			for (int i = 0; i < Height; i++)
			{
				sum += fabs(Elements[i][j]);
			}

			if(sum > max)
			{
				max = sum;
			}
		}

		return max;
	}

	//Matrix second norm
	type Norm2()
	{
		throw exception("Matrix norm 2 not yet Implemented"); // TODO
	}

	//Matrix infinity norm
	type NormInfinity()
	{
		type max = 0;
		for (int i = 0; i < Height; i++)
		{
			type sum = 0;
			for (int j = 0; j < Width; j++)
			{
				sum += fabs(Elements[i][j]);
			}

			if(sum > max)
			{
				max = sum;
			}
		}

		return max;
	}

	//Matrix Frobenius norm
	type NormFrobenius()
	{
		type frob = 0;
		for (int i = 0; i < Height; i++)
		{
			for (int j = 0; j < Width; j++)
			{
				frob += pow(Elements[i][j],2);
			}
		}

		return sqrt(frob);
	}

	//Vector matrix norm
	type VectorMatrixNorm(const Matrix& a)
	{
		if(!a.IsSimetric())
		{
			throw exception("Vector norm could not be found, matrix is asymmetric");
		}
		
		return sqrt((a.MultiplyBy(*this)).VectorMultiplyScalar(*this));  //(Ax,x)^1/2
	}

	#pragma endregion

	#pragma region Check Matrix

	// Check if matrix is simetric
	bool IsSimetric()
	{
		if(Height != Width)
		{
			return false;
		}
		
		for (int i = 0; i < Height - 1; i++)
		{
			for (int j = i+1; j < Width; j++)
			{
				if(Elements[i][j] != Elements[j][i])
				{
					return false;
				}
			}
		}

		return true;
	}

	// Chek if matrix is diagonal
	bool IsDiagonal(int diagonalWidth = 1)
	{
		int shift = abs(diagonalWidth)/2 + 1; //shift from center
		for (int i = 0; i < Height - shift; i++)
		{
			for (int j = i + shift; j < Width; j++)
			{
				if(Elements[i][j] != 0 || Elements[j][i] != 0)
				{
					return false;
				}
			}
		}

		return true;
	}

	// Check if matrix is singular
	bool IsSingular()
	{
		return Determinant() == 0;
	}

	// Check if matrix is positive-definite
	bool IsPositiveDefinite()
	{
		try
		{
			DecompositeLUP(PivotingType::Full);
		}
		catch(exception)
		{
			return false;
		}
		return true;
	}

	// Check if matrix is diagonally dominant
	bool IsDominant()
	{
		for (int i = 0; i < Height; i++)
		{
			type sum = 0;
			for (int j = 0; j < Width; j++)
			{
				if (i != j)
				{
					sum += fabs(Elements[i][j]);
				}
			}
			
			if(sum > fabs(Elements[i][i]))
			{
				return false;
			}
		}

		return true;
	}

	// Check if matrix is strictly diagonally dominant
	bool IsStrictlyDominant()
	{
		for (int i = 0; i < Height; i++)
		{
			type sum = 0;
			for (int j = 0; j < Width; j++)
				if(i != j)
					sum += fabs(Elements[i][j]);
			if(sum >= fabs(Elements[i][i]))
				return false;
		}

		return true;
	}

	// Check if matrix is square matrix
	bool IsSquare()
	{
		return (Height == Width);
	}

	// Check if marix is orthogonal
	bool IsOrthogonal(type epsilon = 0)
	{
		return Transpose().Equals(Inverse(),epsilon);
	}

	// Check if matrix is right stochastic for Markov chain
	bool IsRightStochastic()
	{
		if(Height != Width)
			return false;
		for (int i = 0; i < Height; i++)
		{
			type sum = 0;
			for (int j = 0; j < Width; j++)
			{
				sum += Elements[i][j];
			}
			if(sum != 1)
				return false;
		}
		return true;
	}

	// Check if matrix is left stochastic for Markov chain
	bool IsLeftStochastic()
	{
		if(Height != Width)
			return false;

		for (int j = 0; j < Width; j++)
		{
			type sum = 0;
			for (int i = 0; i < Height; i++)
			{
				sum += Elements[i][j];
			}
			if(sum != 1)
				return false;
		}

		return true;
	}

	// Check if matrix is absorbing for Markov chain
	bool IsAbsorbing(vector<int>& absorbingIndexes, bool& isRightStochastic)
	{
		absorbingIndexes.resize(0);
		if(IsRightStochastic())
		{
			isRightStochastic = true;
			for (int i = 0; i < Height; i++)
			{
				if(Elements[i][i] == 1) //Is absorbing
				{
					type maxRow = 0;
					type maxColumn = 0;
					for (int j = 0; j < Width; j++) //No way out
					{
						if(j != i && Elements[i][j] > maxRow)
							maxRow = Elements[i][j];
					}
					for (int k = 0; k < Height; k++) //It must not be zero (state is reachable)
					{
						if(k != i && Elements[k][i] > maxColumn)
							maxColumn = Elements[k][i];
					}
					if(maxRow == 0 && maxColumn != 0) //It is reachable and can not be leaved
						absorbingIndexes.push_back(i);
				}
			}
			if(absorbingIndexes.size() > 0) //If absorbing was found then true
				return true;
		}
		else if(IsLeftStochastic())
		{
			isRightStochastic = false;
			for (int i = 0; i < Height; i++)
			{
				if(Elements[i][i] == 1) //Is absorbing
				{
					type maxRow = 0;
					type maxColumn = 0;
					for (int j = 0; j < Width; j++) //No way out
					{
						if(j != i && Elements[i][j] > maxRow)
							maxRow = Elements[i][j];
					}
					for (int k = 0; k < Height; k++) //It must not be zero (state is reachable)
					{
						if(k != i && Elements[k][i] > maxColumn)
							maxColumn = Elements[k][i];
					}
					if(maxRow != 0 && maxColumn == 0) //It is reachable and can not be leaved
						absorbingIndexes.push_back(i);
				}
			}
			if(absorbingIndexes.size() > 0) //If absorbing was found then true
				return true;
		}
		return false;
	}

	#pragma endregion

	#pragma region Operators Overload
	//Vector subscript overload
	vector<type>& operator [](const type& index)
	{
		return Elements[index];
	}
	//Sum up elements of two matrixes
	Matrix operator +(const Matrix& b)
	{
		if (Height != b.Height || Width != b.Width)
		{
			throw exception("Matrices with different dimensions can not be added!");
		}
		
		Matrix C(Height, Width);

		for (int i = 0; i < fmin(Height, b.Height); i++)
		{
			for (int j = 0; j < fmin(Width, b.Width); j++)
			{
				C.Elements[i][j] = Elements[i][j] + b.Elements[i][j];
			}
		}

		return C;
	}
	//Substract elements of two matrixes
	Matrix operator -(const Matrix& b)
	{
		if (Height != b.Height || Width != b.Width)
		{
			throw exception("Matrices with different dimensions can not be substracted!");
		}
		
		Matrix C(Height, Width);

		for (int i = 0; i < Height; i++)
		{
			for (int j = 0; j < Width; j++)
			{
				C.Elements[i][j] = Elements[i][j] - b.Elements[i][j];
			}
		}

		return C;
	}
	//Invert matrix elements
	Matrix operator -()
	{
		Matrix C(Height, Width);
		for (int i = 0; i < Height; i++)
		{
			for (int j = 0; j < Width; j++)
			{
				C.Elements[i][j] = -Elements[i][j];
			}
		}
		return C;
	}
	//Uses MultiplyBy method
	Matrix operator *(Matrix& b)
	{
		return MultiplyBy(b);
	}
	//Multiply each matrix element by number
	Matrix operator *(const type& b)
	{
		return MultiplyBy(b);
	}
	//Divide by matrix (multiply by inversed b)
	Matrix operator /(Matrix& b)
	{
		return b.Inverse().MultiplyBy(*this);
	}
	//Divide by number (each element divided by b)
	Matrix operator /(const type& b)
	{
		Matrix retDiv(Height, Width);

		for (int i = 0; i < Height; i++)
		{
			for (int j = 0; j < Width; j++)
			{
				retDiv.Elements[i][j] = Elements[i][j] / b;
			}
		}

		return retDiv;
	}
	//Insert second matrix in first at [0][0]
	void operator =(const Matrix& b)
	{
		Resize(b.Height, b.Width);
		for (int i = 0; i < Height; i++)
		{
			for (int j = 0; j < Width; j++)
			{
				Elements[i][j] = b.Elements[i][j];
			}
		}
	}
	//Set matrix to number on diagonal
	void operator =(const type& b)
	{
		for (int i = 0; i < Height; i++)
		{
			Elements[i][i] = b;
		}
	}
	//If matrix is less then b
	bool operator <(const Matrix& b)
	{
		if(Height != b.Height || Width != b.Width)
		{
			throw exception("Could not compare matrixes. They ar different size");
		}

		return NormInfinity() < b.NormInfinity();
	}
	//If matrix is less or equals b
	bool operator <=(const Matrix& b)
	{
		if(Height != b.Height || Width != b.Width)
		{
			throw exception("Could not compare matrixes. They ar different size");
		}
		
		return NormInfinity() <= b.NormInfinity();
	}
	//If matrix is more then b
	bool operator >(const Matrix& b)
	{
		if(Height != b.Height || Width != b.Width)
		{
			throw exception("Could not compare matrixes. They ar different size");
		}

		return NormInfinity() > b.NormInfinity();
	}
	//If matrix is more then b
	bool operator >=(const Matrix& b)
	{
		if(Height != b.Height || Width != b.Width)
		{
			throw exception("Could not compare matrixes. They ar different size");
		}

		return NormInfinity() >= b.NormInfinity();
	}
	//If matrix is less then number b
	bool operator <(const type& b)
	{
		if(Height != b.Height || Width != b.Width)
		{
			throw exception("Could not compare matrixes. They ar different size");
		}

		return NormInfinity() < b;
	}
	//If matrix is less or equals number b
	bool operator <=(const type& b)
	{
		if(Height != b.Height || Width != b.Width)
		{
			throw exception("Could not compare matrixes. They ar different size");
		}

		return NormInfinity() <= b;
	}
	//If matrix is more then number b
	bool operator >(const type& b)
	{
		if(Height != b.Height || Width != b.Width)
		{
			throw exception("Could not compare matrixes. They ar different size");
		}

		return NormInfinity() > b;
	}
	//If matrix is more then number b
	bool operator >=(const type& b)
	{
		if(Height != b.Height || Width != b.Width)
		{
			throw exception("Could not compare matrixes. They ar different size");
		}

		return NormInfinity() >= b;
	}
	//Checks if element vectors are same
	bool operator ==(const Matrix& b)
	{
		return Equals(b,0);
	}
	//Check is vectors are not the same
	bool operator !=(const Matrix& b)
	{
		return !Equals(b,0);
	}
	// Overloading input trem operator
	friend istream& operator >>(istream& is, Matrix& matrix)
	{
		int setHeight, setWidth;
		is >> setHeight;
		is >> setWidth;

		matrix.Resize(setHeight, setWidth);
		for (int i = 0; i < matrix.GetHeight(); i++)
		{
			for (int j = 0; j < matrix.GetWidth(); j++)
			{
				is >> matrix.Elements[i][j];//scanf("%lf",&Elements[i][j]);
			}
		}

		return is;
	}
	// Overloading output strem operator
	friend ostream& operator <<(ostream& os, const Matrix& matrix)
	{
		os << matrix.GetHeight() << " " << matrix.GetWidth() << endl;
		// Print matrix
		for (int i = 0; i < matrix.GetHeight(); i++)
		{
			for (int j = 0; j < matrix.GetWidth(); j++)
			{
				os << fixed << matrix.Elements[i][j] << "\t";//cout<<Elements[i][j]<<"\t";
			}
			os << endl;
		}

		return os;
	}
	
	#pragma endregion

    #pragma region Static Methods
	//Generate random T matrix //NOT IMPLEMENTED
	static Matrix Random(int height, int width, int min = -RAND_MAX, int max = RAND_MAX)
	{
		Matrix random(height,width);
		uniform_real_distribution<type> unif(min,max);
		random_device rand;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				random.Elements[i][j] = unif(rand);
			}
		}
		return random;
	}
	//Static Hilbert matrix
	static Matrix Hilbert(int size)
	{
		Matrix hilbert(size,size);
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				hilbert.Elements[i][j] = 1.0/(i+j+1);
			}
		}
		return hilbert;
	}
	//Generate Vandermonde matrix
	static Matrix Vandermonde(const vector<type> &x)
	{
		Matrix vandermonde(x.size(),x.size());
		for (int i = 0; i < vandermonde.Height; i++)
		{
			for (int j = 0; j < vandermonde.Width; j++)
			{
				vandermonde.Elements[i][j] = pow(x[i],j);
			}
		}
		return vandermonde;
	}
	//Generate Sylvester matrix
	static Matrix Sylvester(const vector<type>& p, const vector<type>& q)
	{
		int pSize = p.size(), qSize = q.size();
		int m = pSize - 1;
		int n = qSize - 1;
		Matrix sylvester(m + n, m + n);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < pSize; j++)
			{
				sylvester.Elements[i][j + i] = p[pSize - j - 1];
			}
		}
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < qSize; j++)
			{
				sylvester.Elements[i + n][j + i] = q[qSize - j - 1];
			}
		}
		return sylvester;
	}
	//Generate Lehmer matrix
	static Matrix Lehmer(int height, int width)
	{
		Matrix lehmer(height, width);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if(j < i)
					lehmer.Elements[i][j] = double(j+1) / (i+1);
				else
					lehmer.Elements[i][j] = double(i+1) / (j+1);
			}
		}
		return lehmer;
	}
	//Generate Redheffer matrix
	static Matrix Redheffer(int size)
	{
		Matrix redheffer(size, size);
		for (int i = 0; i < redheffer.Height; i++)
		{
			for (int j = 0; j < redheffer.Width; j++)
			{
				if((j+1) % (i+1) == 0 || j == 0)
					redheffer.Elements[i][j] = 1;
				else
					redheffer.Elements[i][j] = 0;
			}
		}
		return redheffer;
	}
	//Static diagonal matrix
	static Matrix Diagonal(int height, int width, type diagValue = 1)
	{
		Matrix diagonal(height, width);
		for (int i = 0; i < fmin(height, width); i++)
		{
			diagonal.Elements[i][i] = diagValue;
		}
		return diagonal;
	}
	//Static diagonal matrix with vector values
	static Matrix Diagonal(const vector<type>& diagonalElements)
	{
		int size = diagonalElements.size();
		Matrix diagonal(size, size);
		for (int i = 0; i < size; i++)
		{
			diagonal.Elements[i][i] = diagonalElements[i];
		}
		return diagonal;
	}
	//Generate counterdiagonal matrix
	static Matrix CounterDiagonal(int size, type diagValue = 1)
	{
		Matrix counterDiagonal(size,size);
		for (int i = 0; i < size; i++)
		{
			counterDiagonal.Elements[i][size-i-1] = diagValue;
		}
		return counterDiagonal;
	}
	//Returns values of interpolating polynomial
	static Matrix Interpolate(const vector<type>& x, const vector<type>& y)
	{
		Matrix yMatrix(y.size(),1);
		for (int i = 0; i < yMatrix.Height; i++)
		{
			yMatrix.Elements[i][0] = y[i];
		}
		return Vandermonde(x).SolveDirectGauss(yMatrix);
	}
	//Rotation matrix in 2D
	static Matrix Rotation2D(double angle, bool radian = false)
	{
		Matrix rotation(2,2);
		if(!radian) // pi/180 - degree to radian
			angle *= 0.01745329251; 
		double sine = sin(angle);
		double cosine = cos(angle);
		rotation.Elements[0][0] = cosine;
		rotation.Elements[0][1] = -sine;
		rotation.Elements[1][0] = sine;
		rotation.Elements[1][1] = cosine;
		return rotation;
	}
	
	#pragma endregion

	#pragma region Read/Write
	//Read from file
	void ReadFromFile(char *fileName)
	{
		ifstream in(fileName);
		try
		{
			int height,width;
			in>>height>>width;
			Resize(height,width);
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					in>>Elements[i][j];//scanf("%lf",&Elements[i][j]);
				}
			}
			in.close();
		}
		catch(exception e)
		{
			in.close();
			throw exception("Could not read matrix from file");
		}		
	}
	//Write to file
	void WriteToFile(char *fileName,int pointPrecision = 6) const
	{
		ofstream out(fileName);
		out<<Height<<" "<<Width<<endl;
		out.precision(pointPrecision);
		for (int i = 0; i < Height; i++)
		{
			for (int j = 0; j < Width; j++)
			{
				out<<fixed<<Elements[i][j]<<"\t";//cout<<Elements[i][j]<<"\t";
			}
			out<<endl;
		}
		out.close();
	}

	#pragma endregion

	#pragma region Private Methods
private:
	//Find absolute value of a template typename
	type fabs(type a)
	{
		return a<0 ? -a:a;
	}

	#pragma endregion

};