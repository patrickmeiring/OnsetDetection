using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection
{
    static class DiscreteFourierTransform
    {
        /// <summary>
        /// Performs a Discrete Fourier Transform on the specified inputs.
        /// </summary>
        /// <param name="real">The real-component to transform.</param>
        /// <param name="imaginary">The imaginary component resulting from the transform.</param>
        public static void Transform(double[] real, double[] imaginary)
        {
            Debug.Assert(real.Length == imaginary.Length);
            Array.Clear(imaginary, 0, imaginary.Length);

            ApplyHanningWindow(real);
            Interlace(real);

            int magnitude = (int)Math.Log(real.Length, 2.0);
            for (int m = 0; m < magnitude; m++)
            {
                int le2 = 1 << m;
                int le = le2 * 2;
                double ur = 1.0;
                double ui = 0.0;

                for (int j = 0; j < le2; j++)
                {
                    for (int i = j; i < real.Length; i += le)
                    {
                        int ip = i + le2;
                        double realTemp = real[ip] * ur - imaginary[ip] * ui;
                        double imaginaryTemp = real[ip] * ui + imaginary[ip] * ur;
                        real[ip] = real[i] - realTemp;
                        imaginary[ip] = imaginary[i] - imaginaryTemp;
                        real[i] += realTemp;
                        imaginary[i] += imaginaryTemp;
                    }

                    double realSin = Math.Cos(Math.PI / le2);
                    double imaginarySin = -Math.Sin(Math.PI / le2);

                    double oldUReal = ur;
                    double oldIReal = ui;
                    ur = oldUReal * realSin - oldIReal * imaginarySin;
                    ui = oldUReal * imaginarySin + oldIReal * realSin;
                }
            }
        }

        public static void KeepRealComponent(double[] real, double[] imaginary)
        {
            for (int i = 0; i < real.Length; i++)
            {
                double realTemp = real[i];
                double imaginaryTemp = imaginary[i];
                real[i] = Math.Sqrt(realTemp * realTemp + imaginaryTemp * imaginaryTemp);
            }
        }

        /// <summary>
        /// Performs an in-place interlace of the specified input to prepare it for
        /// the Discrete Fourier Transform.
        /// </summary>
        /// <param name="input">The inputs to interlace.</param>
        private static void Interlace(double[] input)
        {
            int halfLength = input.Length / 2;

            int j = halfLength;
            for (int i = 1; i < input.Length - 1; i++)
            {
                if (i < j)
                {
                    double tempReal = input[j];
                    input[j] = input[i];
                    input[i] = tempReal;
                }

                // Reverse add/carry routine.
                int k = halfLength; // "1" in the reversed number.

                while (k <= j) // If the bit we're looking at is already set
                {
                    j -= k; // Unset it
                    k /= 2; // Do a "carry", smaller is bigger in reversed number.
                }
                j += k; // Perform the add of 1, or overflow.
            }
        }

        private static void ApplyHanningWindow(double[] input)
        {
            double alpha = 0.50; // 0.54 for Hamming
            double beta = 1 - alpha;

            double nCoefficient = Math.PI * 2.0 / (input.Length - 1);

            for (int i = 0; i < input.Length; i++)
            {
                double hammingFactor = alpha - beta * Math.Cos(nCoefficient * i);
                input[i] *= hammingFactor;
            }
        }

    }
}
