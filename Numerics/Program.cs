using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Numerics
{
    class Program
    {
        static void Main(string[] args)
        {
            Matrix A = new Matrix(3, 3);
            A.Elements = new double[,]
                             {
                                 {1, 2, 3},
                                 {1, 1, 2},
                                 {1, 3, 3}
                             };

            Matrix B = new Matrix(3, 3);
            B.Elements = new double[,]
                             {
                                 {1, 0, 0},
                                 {0, 1, 0},
                                 {0, 0, 1}
                             };

            Matrix C = A.DecompositeLu();
                
            Console.WriteLine(A);
            Console.WriteLine(C);
        }
    }
}
