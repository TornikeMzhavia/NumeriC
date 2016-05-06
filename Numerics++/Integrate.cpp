using namespace std;

class integrate
{
public:
	static double a;
	static double b;
	static int n;

	static double Square(double (*function)(double x))
	{
		double h,sum;
		h = (a-b)/n;
		sum = (*function)(a) + (*function)(b);
		for (int i = 1; i <= n; i++)
		{
			sum += (*function)(a + i*h + h/2);
		}
		return sum * h;
	}

	static double Trap(double (*function)(double x))
	{
		double h,sum;
		h = (a-b)/n;
		sum = (*function)(a) + (*function)(b);
		for (int i = 1; i <= n; i++)
		{
			sum += 2 * (*function)(a + i*h);
		}
		return sum * h / 2;
	}

	static double Simpson(double (*function)(double x))
	{
		double h,sum;
		h = (a-b)/n;
		sum = (*function)(a) + (*function)(b);
		for (int i = 1; i <= n; i+=2)
		{
			sum += 4 * (*function)(a + i*h);
		}
		for (int i = 2; i <= n-1; i+=2)
		{
			sum += 2 * (*function)(a + i*h);
		}
		return sum * h / 3;
	}
};