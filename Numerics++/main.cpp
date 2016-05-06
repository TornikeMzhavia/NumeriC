#include <iostream>
#include <time.h>
#include <cmath>
#include <ccomplex>
#include "Numeric.h"
using namespace std;

void DoMarkov()
{
	Matrix<> T("T.txt");// = Matrix<double>::Random(17,17);
	//A.WriteToFile("A.txt");
	//(A * A.Inverse()).WriteToFile("CON");

	Matrix<> S("S.txt");//("B.txt");// = Matrix<double>::Random(3,2);
	//(A.MultiplyBy(x) - B).WriteToFile("huge.txt",28);
	//B.WriteToFile("B.txt");
	int n;
	cin >> n;
	S = S*T.Power(n);
	for (int i = 0; i < S.GetWidth(); i++)
	{
		cout << i << " - " << S.Elements[0][i] << endl;
	}
}

void SolveSportsBetting()
{
	int n;
	vector<double> odds;
	double amount = 1.0;

	cout << "How many bets are there?" << endl;
	cin >> n;

	odds.resize(n);

	cout << "Enter them" << endl;
	for (int i = 0; i < n; i++)
	{
		cin >> odds[i];
	}

	//cout<<"How much money do you have?"<<endl;
	//cin>>amount;

#pragma region Matrix initialization

	Matrix<> A(n, n);
	Matrix<> B(n, 1);

	for (int i = 0; i < A.GetHeight() - 1; i++)
	{
		for (int j = 0; j < A.GetWidth(); j++)
		{
			if (j == 0)
			{
				A[i][j] = odds[j];
			}
			else if (j == i + 1)
			{
				A[i][j] = -odds[j];
			}
		}
	}

	for (int j = 0; j < A.GetWidth(); j++)
	{
		A[A.GetHeight() - 1][j] = 1;
	}

	B[B.GetHeight() - 1][0] = amount;

#pragma endregion

	double oddsKey = 0;
	for (int i = 0; i < odds.size(); i++)
	{
		oddsKey += 100 / odds[i];
	}

	Matrix<> stakes = A.SolveDirectGauss(B);
	cout << endl << stakes << endl;
	cout << "Guaranteed Winning: " << stakes.Elements[0][0] * A.Elements[0][0] - (B.Elements[B.GetHeight() - 1][B.GetWidth() - 1]) << endl;
	cout << "OddsKey: " << oddsKey << endl;
}

long long GetFibonacciNumber(int n)
{
	Matrix<long long> A(2, 2);
	A[0][0] = 1;
	A[0][1] = 1;
	A[1][0] = 1;
	A[1][1] = 0;

	return A.Power(n).Elements[0][0];
}

void main()
{
	try
	{
		/*while(true)
		{
			SolveSportsBetting();
		}*/

		Matrix<double> A("A.txt");// = Matrix<>::Random(32, 32, 0, 1);
		Matrix<double> B("B.txt");// = Matrix<>::Random(32, 32, 0, 1);

		int start = clock();

		//cout << A.Determinant() << endl;
		//Matrix<> stdMult = A.MultiplyBy(B);
		//Matrix<> strassenMult = A.MultiplyByStrassen(B,32);
		//bool a = stdMult.Equals(strassenMult, 0.01);
		//cout << a << endl;
		for (int i = 0; i < 1000; i++)
		{
			//A.LearnFromData(B);
			//.DecompositeLUP(PivotingType::Full);//(PivotingType::None);
			//A.SolveIterativeSOR(B,0.001,1.25);
			//A.Transpose();
			//A.MultiplyBy(B);
			//A.Determinant();
			//A.Inverse();
			//A.ConditionNumber();
			//A.SolveDirectThomas(B);
			//A.Power(100);
		}

		cout << clock() - start << endl;
	}
	catch (exception e)
	{
		cout << "Error occured!\n" << e.what() << endl;
		system("pause");
	}
}