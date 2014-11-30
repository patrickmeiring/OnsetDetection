using OnsetDetection.Testing;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection.NeuralNetworks
{
    class RecurrentLayer : Layer
    {
        private readonly double[] biasWeights;
        private readonly double[,] inputWeights;
        private readonly double[,] internalWeights;

        public RecurrentLayer(NetworkConfiguration configuration, int size, int inputSize) 
            : base(configuration, size, inputSize)
        {
            this.biasWeights = new double[size];
            Randomise(biasWeights, inputSize + size + 1);

            this.inputWeights = new double[inputSize, size];
            Randomise(inputWeights, inputSize + size + 1);

            this.internalWeights = new double[size, size];
            Randomise(internalWeights, inputSize + size + 1);
        }

        public void FeedForward(RecurrentState state)
        {
            double[] weightedSums = state.WeightedSums;

            Array.Clear(weightedSums, 0, weightedSums.Length);
            Sum(biasWeights, weightedSums);
            WeightedInputSum(state.Inputs, inputWeights, weightedSums);
            WeightedInputSum(state.LastOutputs, internalWeights, weightedSums);
            OutputFromActivation(weightedSums, state.Outputs);
        }

        public void BackPropogate(RecurrentTrainingState state)
        {
            Debug.Assert(state.Now.WeightedSums != state.Now.Outputs);

            // Initially, state.Errors contains just those backpropogated from the next layer.
            // These are d(E)/d(outputs of this layer).

            // Add the error from the timestep that was next in time.
            WeightedOutputSum(state.InternalErrors, internalWeights, state.Errors);

            // Multiply by the derivative of the activation function to finalise
            // errors with respect to the weighted sum of the unit for this layer.
            MultiplyByActivationDerivative(state.Now.WeightedSums, state.Errors);
            
            // Propogate the errors back to the input.
            Array.Clear(state.InputErrors, 0, state.InputErrors.Length);
            WeightedOutputSum(state.Errors, inputWeights, state.InputErrors);

            // The errors for this layer become the errors at the timestep T+1
            // as we move to the previous timestep.
            Array.Copy(state.Errors, state.InternalErrors, state.Errors.Length);

            // Add the error of each weight. This is its value of the input it connects to
            // times the error of the output it connects to.
            Sum(state.Errors, state.BiasWeightErrors);
            SumProducts(state.Now.Inputs, state.Errors, state.InputWeightErrors);
            SumProducts(state.Last.Outputs, state.Errors, state.InternalWeightErrors);
        }

        public void ApplyWeightChanges(RecurrentTrainingState state, double learningCoefficient)
        {
            Layer.ApplyWeightChanges(biasWeights, state.BiasWeightErrors, learningCoefficient);
            Layer.ApplyWeightChanges(inputWeights, state.InputWeightErrors, learningCoefficient);
            Layer.ApplyWeightChanges(internalWeights, state.InternalWeightErrors, learningCoefficient);
        }

        public override double SumAbsoluteWeight
        {
            get { return Layer.SumAbsolute(inputWeights) + Layer.SumAbsolute(internalWeights) + Layer.SumAbsolute(biasWeights); }
        }

        public override double WeightCount
        {
            get { return InputSize * Size + Size * Size + Size; }
        }
    }
}
