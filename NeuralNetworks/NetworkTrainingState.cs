using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection
{
    class NetworkTrainingState
    {
        public readonly RecurrentNetwork Network;
        private List<object> states;

        public readonly double[] Errors;

        public NetworkTrainingState(RecurrentNetwork network)
        {
            Network = network;
            double[] inputErrors = new double[network.Layers[0].InputSize];
            double[] errors = null;

            states = new List<object>();
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

        public void BackPropogate(NetworkState last, NetworkState now)
        {
            for (int i = states.Count - 1; i >= 0; i--)
            {
                object state = states[i];
                Layer layer = Network.Layers[i];

                RecurrentTrainingState recurrantState = state as RecurrentTrainingState;
                FeedForwardTrainingState feedForwardState = state as FeedForwardTrainingState;
                if (recurrantState != null)
                {
                    recurrantState.Last = (RecurrentState)last.states[i];
                    recurrantState.Now = (RecurrentState)now.states[i];
                    ((RecurrentLayer)layer).BackPropogate(recurrantState);
                }
                else
                {
                    feedForwardState.Now = (FeedForwardState)now.states[i];
                    ((FeedForwardLayer)layer).BackPropogate(feedForwardState);
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
