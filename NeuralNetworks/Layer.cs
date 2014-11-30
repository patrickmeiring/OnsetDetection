using OnsetDetection.Testing;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection.NeuralNetworks
{
    abstract class Layer
    {
        public readonly int Size;
        public readonly int InputSize;
        private readonly NetworkConfiguration configuration;

        public Layer(NetworkConfiguration configuration, int size, int inputSize)
        {
            this.configuration = configuration;
            this.Size = size;
            this.InputSize = inputSize;
        }

        public abstract double SumAbsoluteWeight { get; }

        public abstract double WeightCount { get; }

        protected virtual double Activation(double x)
        {
//            return tanh(x);
            return 1.7159 * tanh((2.0 / 3.0) * x);
        }

        private static double tanh(double x)
        {
            double ePow2x = Math.Exp(2 * x);
            return (ePow2x - 1) / (ePow2x + 1);
        }

        private static double sech(double x)
        {
            double ePowx = Math.Exp(x);
            return 2.0 / (ePowx + (1.0 / ePowx));
        }

        protected virtual double ActivationDerivative(double x)
        {
     //       derivative for 1.7159 * tanh(2.0/3.0)
            double s = sech((2.0/3.0) * x);
            return (2.0 / 3.0) * 1.7159 * s * s;
          // double s = sech(x);
         //  return s * s;
        }

        protected void OutputFromActivation(double[] inputs, double[] outputs)
        {
            Debug.Assert(inputs.Length == outputs.Length);

            for (int i = 0; i < inputs.Length; i++)
            {
                //Debug.Assert(!double.IsNaN(inputs[i]));
                double activated = Activation(inputs[i]);
                if (double.IsNaN(activated))
                    throw new ArgumentException("NaN hit");
//                Debug.Assert(!double.IsNaN(activated));
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

        protected void Randomise(double[,] weights, int fanIn)
        {
            double fanInDouble = (double)fanIn;
            for (int x = 0; x < weights.GetLength(0); x++)
            {
                for (int y = 0; y < weights.GetLength(1); y++)
                {
//                    double normal = RandomNormal();
                    weights[x, y] = Random(fanIn); // (random.NextDouble() - 0.5) * (4.8 / fanInDouble);// normal * 0.01; //(random.NextDouble() - 0.5) * (4.8 / fanInDouble);//normal * 0.2 - 0.1; 
                }
            }
        }

        protected void Randomise(double[] weights, int fanIn)
        {
            double fanInDouble = (double)fanIn;
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = Random(fanIn);// (random.NextDouble() - 0.5) * (4.8 / fanInDouble); //
            }
        }

        private double Random(double fanIn)
        {
            if (configuration.WeightInitialisationMethod == RandomType.Linear)
            {
                return (configuration.NextRandom() - 0.5) * (4.8 / fanIn) * configuration.WeightInitialisationSize;
            }
            else if (configuration.WeightInitialisationMethod == RandomType.Guassian)
            {
                return RandomNormal() * configuration.WeightInitialisationSize;
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Generates normally distributed random numbers, with a mean of zero and variance of one.
        /// </summary>
        /// <returns></returns>
        private double RandomNormal()
        {
            double uniform1 = configuration.NextRandom();
            double uniform2 = configuration.NextRandom();

            // Use Box-Muller transform to generate normally distributed random numbers.
            return Math.Sqrt(-2.0 * Math.Log(uniform1)) * Math.Cos(2.0 * Math.PI * uniform2);
        }

        protected static double SumAbsolute(double[,] weights)
        {
            double sum = 0.0;
            for (int x = 0; x < weights.GetLength(0); x++)
            {
                for (int y = 0; y < weights.GetLength(1); y++)
                {
                    sum += Math.Abs(weights[x, y]);
                }
            }
            return sum;
        }

        protected static double SumAbsolute(double[] weights)
        {
            return weights.Sum(weight => Math.Abs(weight));
        }
    }
}
