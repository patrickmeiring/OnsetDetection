using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection
{
    class Layer
    {
        public readonly int Size;
        public readonly int InputSize;

        public Layer(int size, int inputSize)
        {
            this.Size = size;
            this.InputSize = inputSize;
        }

        protected virtual double Activation(double x)
        {
            // tanh
            double ePow2x = Math.Exp(2 * x);
            return (ePow2x - 1) / (ePow2x + 1);
        }

        protected virtual double ActivationDerivative(double x)
        {
            // derivative for tanh
            double tanh = Activation(x);
            return 1 - tanh * tanh;
        }

        protected void OutputFromActivation(double[] inputs, double[] outputs)
        {
            Debug.Assert(inputs.Length == outputs.Length);

            for (int i = 0; i < inputs.Length; i++)
            {
                Debug.Assert(!double.IsNaN(inputs[i]));
                double activated = Activation(inputs[i]);
                Debug.Assert(!double.IsNaN(activated));
                outputs[i] = activated;
            }
        }

        protected void MultiplyByActivationDerivative(double[] inputs, double[] errors)
        {
            Debug.Assert(inputs.Length == errors.Length);

            for (int i = 0; i < inputs.Length; i++)
            {
                errors[i] *= ActivationDerivative(inputs[i]);
            }
        }

        protected static void Sum(double[] weights, double[] outputs)
        {
            Debug.Assert(weights.Length == outputs.Length);
            
            for (int o = 0; o < outputs.Length; o++)
            {
                double sum = outputs[o];
                sum += weights[o];
                outputs[o] = sum;
            }
        }

        protected static void WeightedInputSum(double[] inputs, double[,] weights, double[] outputs)
        {
            Debug.Assert(weights.GetLength(0) == inputs.Length);
            Debug.Assert(weights.GetLength(1) == outputs.Length);

            for (int o = 0; o < outputs.Length; o++)
            {
                double sum = outputs[o];
                for (int i = 0; i < inputs.Length; i++)
                {
                    sum += inputs[i] * weights[i, o];
                }
                outputs[o] = sum;
            }
        }

        protected static void WeightedOutputSum(double[] outputs, double[,] weights, double[] inputs)
        {
            Debug.Assert(weights.GetLength(0) == inputs.Length);
            Debug.Assert(weights.GetLength(1) == outputs.Length);

            for (int i = 0; i < inputs.Length; i++)
            {
                double sum = inputs[i];
                for (int o = 0; o < outputs.Length; o++)
                {
                    sum += outputs[o] * weights[i, o];
                }
                inputs[i] = sum;
            }
        }

        protected static void SumProducts(double[] inputs, double[] outputs, double[,] weights)
        {
            Debug.Assert(weights.GetLength(0) == inputs.Length);
            Debug.Assert(weights.GetLength(1) == outputs.Length);

            for (int i = 0; i < inputs.Length; i++)
            {
                for (int o = 0; o < outputs.Length; o++)
                {
                    weights[i, o] += inputs[i] * outputs[o];
                }
            }
        }

        protected static void ApplyWeightChanges(double[] weights, double[] errors, double learningCoefficient)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] -= errors[i] * learningCoefficient;
            }
        }

        protected static void ApplyWeightChanges(double[,] weights, double[,] errors, double learningCoefficient)
        {
            for (int x = 0; x < weights.GetLength(0); x++)
            {
                for (int y = 0; y < weights.GetLength(1); y++)
                {
                    weights[x, y] -= errors[x, y] * learningCoefficient;
                }
            }
        }

        [ThreadStatic]
        protected static Random random = new Random(0);
        protected static void Randomise(double[,] weights, int fanIn)
        {
            double fanInDouble = (double)fanIn;
            for (int x = 0; x < weights.GetLength(0); x++)
            {
                for (int y = 0; y < weights.GetLength(1); y++)
                {
                    weights[x, y] = (random.NextDouble() - 0.5) * (4.8 / fanInDouble);
                }
            }
        }

        protected static void Randomise(double[] weights, int fanIn)
        {
            double fanInDouble = (double)fanIn;
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = (random.NextDouble() - 0.5) * (4.8 / fanInDouble);
            }
        }
    }
}
