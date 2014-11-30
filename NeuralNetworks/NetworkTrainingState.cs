using OnsetDetection.NeuralNetworks;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection.NeuralNetworks
{
    class NetworkTrainingState
    {
        public readonly RecurrentNetwork Network;
        private List<TrainingState> states;

        public readonly double[] Errors;

        public NetworkTrainingState(RecurrentNetwork network)
        {
            Network = network;
            double[] inputErrors = new double[network.Layers[0].InputSize];
            double[] errors = null;

            states = new List<TrainingState>();
            for (int i = 0; i < network.Layers.Count; i++)
            {
                Layer layer = network.Layers[i];
                errors = new double[layer.Size];

                if (layer is RecurrentLayer)
                {
                    //RecurrentState lastState = last != null ? (RecurrentState)last.states[i] : null;
                    states.Add(new RecurrentTrainingState(inputErrors, errors));
                }
                else
                {
                    states.Add(new FeedForwardTrainingState(inputErrors, errors));
                }

                // The input errors next layer are the errors of the last.
                inputErrors = errors;
            }
            this.Errors = errors;
        }

        public void Prepare(double momentum)
        {
            foreach (TrainingState state in states)
            {
                // Keep a proportion of the the total error backpropogating the last sample.
                // This functions as a momentum term.
                state.MultiplyError(momentum);
            }
        }

        public void BackPropogate(NetworkState last, NetworkState now)
        {
            for (int i = states.Count - 1; i >= 0; i--)
            {
                object state = states[i];
                Layer layer = Network.Layers[i];

                RecurrentLayer recurrentLayer = layer as RecurrentLayer;
                FeedForwardLayer feedForwardLayer = layer as FeedForwardLayer;
               
                if (recurrentLayer != null)
                {
                    RecurrentTrainingState recurrentState = (RecurrentTrainingState)state;
                    recurrentState.Last = (RecurrentState)last.states[i];
                    recurrentState.Now = (RecurrentState)now.states[i];
                    recurrentLayer.BackPropogate(recurrentState);
                }
                else
                {
                    FeedForwardTrainingState feedForwardState = (FeedForwardTrainingState)state;
                    feedForwardState.Now = (FeedForwardState)now.states[i];
                    feedForwardLayer.BackPropogate(feedForwardState);
                }
            }
        }

        public void ApplyWeightChanges(double learningCoefficient)
        {
            for (int i = 0; i < states.Count; i++)
            {
                object state = states[i];
                Layer layer = Network.Layers[i];

                RecurrentTrainingState recurrantState = state as RecurrentTrainingState;
                FeedForwardTrainingState feedForwardState = state as FeedForwardTrainingState;
                if (recurrantState != null)
                {
                    ((RecurrentLayer)layer).ApplyWeightChanges(recurrantState, learningCoefficient);
                }
                else
                {
                    ((FeedForwardLayer)layer).ApplyWeightChanges(feedForwardState, learningCoefficient);
                }
            }
        }
    }
}
