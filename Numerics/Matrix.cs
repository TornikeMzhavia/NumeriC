using System;

namespace Numerics
{
    class Matrix
    {
        public double[,] Elements;
        private int Height { get; set; }
        private int Width { get; set; }

        #region Consructors

        public Matrix(int height = 0,int width = 0)
        {
            Height = height;
            Width = width;
            Elements = new double[height, width];
        }

        #endregion

        #region Methods

        public int GetHeight()
        {
            return Height;
        }

        public int GetWidth()
        {
            return Width;
        }

        public Matrix MultiplyBy(Matrix bMatrix)
        {
            Matrix retMultiplied = new Matrix(this.Height,bMatrix.Width);

            if(Width == bMatrix.Height)
            {
                //Multiply two matrixes 
                for (int i = 0; i < retMultiplied.Height; i++)
                {
                    for (int j = 0; j < retMultiplied.Width; j++)
                    {
                        double sum = 0;
                        for (int k = 0; k < Width; k++)
                        {
                            sum += Elements[i, k]*bMatrix.Elements[k, j];
                        }

                        retMultiplied.Elements[i, j] = sum;
                    }
                }
            }
            else
            {
                throw new Exception("Matrixes can not by multiplied. Width of the first is not the same as seconds height.");
            }

            return retMultiplied;
        }

        public Matrix MultiplyBy(double number)
        {
            Matrix retMultiplied = new Matrix(Height, Width);

            for (int i = 0; i < Height; i++)
            {
                for (int j = 0; j < Width; j++)
                {
                    retMultiplied.Elements[i, j] = Elements[i, j]*number;
                }
            }

            return retMultiplied;
        }

        public Matrix DecompositeLu()
        {
            Matrix retComposite = new Matrix(Height, Width);
            //copy
            for (int i = 0; i < Height; i++)
            {
                for (int j = 0; j < Width; j++)
                {
                    retComposite.Elements[i, j] = Elements[i, j];
                }
            }
            //compute
            if (Height == Width)
            {
                for (int j = 0; j < Width; j++)
                {
                    for (int i = j + 1; i < Height; i++)
                    {
                        double k = -(Elements[i,j]/Elements[j,j]);
                        for (int c = j; c < Width; c++)
                        {
                            retComposite.Elements[i, c] += Elements[j,c]*k; //U part
                        }
                        retComposite.Elements[i,j] = k; //L part
                    }
                }
            }
            else
            {
                throw new Exception("Matrix can not be decomposited. It is not squared matrix.");
            }

            return retComposite;
        }

        #region Overload
        public static Matrix operator +(Matrix a,Matrix b)
        {
            Matrix C = new Matrix(a.Height, a.Width);

            if(a.Height == b.Height && a.Width == b.Width)
            {
                for (int i = 0; i < C.Height; i++)
                {
                    for (int j = 0; j < C.Width; j++)
                    {
                        C.Elements[i, j] = a.Elements[i, j] + b.Elements[i, j];
                    }
                }
            }
            else
            {
                throw new Exception("Matrixes could not be added up, theys sizes are different");
            }

            return C;
        }

        public static Matrix operator -(Matrix a, Matrix b)
        {
            Matrix C = new Matrix(a.Height, a.Width);

            if (a.Height == b.Height && a.Width == b.Width)
            {
                for (int i = 0; i < C.Height; i++)
                {
                    for (int j = 0; j < C.Width; j++)
                    {
                        C.Elements[i, j] = a.Elements[i, j] - b.Elements[i, j];
                    }
                }
            }
            else
            {
                throw new Exception("Matrixes could not be added up, theys sizes are different");
            }

            return C;
        }

        public static bool operator ==(Matrix a, Matrix b)
        {
            if(a != null && b != null)
            {
                return a.Elements.Equals(b.Elements);
            }
            return false;
        }

        public static bool operator !=(Matrix a, Matrix b)
        {
            if (a != null && b != null)
            {
                return !a.Elements.Equals(b.Elements);
            }
            return true;
        }

        public override string ToString()
        {
            string retMatrixString = String.Empty;
            for (int i = 0; i < Height; i++)
            {
                for (int j = 0; j < Width; j++)
                {
                    retMatrixString += String.Format("{0}\t",Elements[i,j]);
                }
                retMatrixString += "\n";
            }
            
            return retMatrixString;
        }
        #endregion

        #endregion

    }
}
