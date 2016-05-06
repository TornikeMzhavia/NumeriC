using namespace std;

#define PI 3.1415926535897932384626433832795

class Approximation
{
public:
	vector<double> Xs;
	vector<double> Ys;
	
	double Lagrange(double x,double aInterval, double bInterval, int pointNumber,double (*function)(double x), bool chebishev = false)
	{
		SetInput(aInterval, bInterval, pointNumber,&(*function),chebishev);
		double approximation= 0;
		for (int i = 0; i < pointNumber; i++)
		{
			approximation += Ys[i] * LagrangeBase(x,i,pointNumber);
		}
		return approximation;
	}

	double LagrangeBase(double x,int index,int pointNumber)
	{
		double lagrangian = 1;
		for (int i = 0; i < pointNumber; i++)
		{
			if(i != index)
			{
				lagrangian *= (x - Xs[i])/(Xs[index] - Xs[i]);
			}
		}
		return lagrangian;
	}

	double Newton(double x,double aInterval, double bInterval, int pointNumber,double (*function)(double x), bool chebishev = false)
	{
		SetInput(aInterval, bInterval, pointNumber,&(*function),chebishev);
		vector<double> divDiff = DividedDifference(x,pointNumber);
		double approximation= 0;
		for (int i = 0; i < pointNumber; i++)
		{
			approximation += divDiff[i] * NewtonBase(x,i);
		}
		return approximation;
	}

	double NewtonBase(double x,int index)
	{
		double newtonBase = 1;
		for (int i = 0; i < index; i++)
		{
			newtonBase *= x - Xs[i];
		}
		return newtonBase;
	}

	vector<double> DividedDifference(double x,int pointNumber)
	{
		vector<vector <double> > dividedDifference(pointNumber);
		for (int i = 0; i < pointNumber; i++)
		{
			dividedDifference[i].resize(pointNumber - i);
			dividedDifference[i][0] = Ys[i];
		}
		for (int j = 1; j < pointNumber; j++)
		{
			for (int i = 0; i < pointNumber - j; i++) 
			{
				dividedDifference[i][j] = (dividedDifference[i + 1][j - 1] - dividedDifference[i][j - 1]) / (Xs[i + j] - Xs[i]);
			}
		}
		return dividedDifference[0];
	}

	static double RungeFunction(double x)
	{
		return 1.0/(1 + 25*x*x);
	}

private:

	void SetInput(double aInterval, double bInterval, int pointNumber,double (*function)(double x), bool chebishev = false)
	{
		Xs.resize(pointNumber);
		Ys.resize(pointNumber);
		if(!chebishev)
		{
			double step = (bInterval-aInterval)/(pointNumber - 1);
			for (int i = 0; i < pointNumber; i++)
			{
				Xs[i] = aInterval + i * step;
				Ys[i] = (*function)(Xs[i]);
			}
		}
		else
		{
			double step = (bInterval + aInterval)/2.0;
			for (int i = 0; i < pointNumber; i++)
			{
				Xs[i] = step + ((bInterval - aInterval) * cos((2*i+1)*PI/2.0))/2.0;
				Ys[i] = (*function)(Xs[i]);
			}
		}
	}
};