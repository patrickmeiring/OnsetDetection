using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection.NeuralNetworks
{
    class NetworkState
    {
        public readonly RecurrentNetwork Network;
        internal List<object> states;
        public readonly double[] Input;
        public readonly double[] Output;

        public NetworkState(RecurrentNetwork network) 
            : this(network, null)
        {
        }

        public NetworkState(RecurrentNetwork network, NetworkState last)
        {
            this.Network = network;
            states = new List<object>();

            double[] input = new double[network.Layers[0].InputSize];
            this.Input = input;

            double[] output = null;
            for (int i = 0; i < network.Layers.Count; i++)
            {
                Layer layer = network.Layers[i];
                output = new double[layer.Size];

                if (layer is RecurrentLayer)
                {
                    RecurrentState lastState = last != null ? (RecurrentState)last.states[i] : null;
                    states.Add(new RecurrentState(input, output, lastState));
                }
                else
                {
                    states.Add(new FeedForwardState(input, output));
                }

                // The input of the next layer is the output of the last.
                input = output;
            }
            this.Output = output;
        }

        public void FeedForward()
        {
            for (int i = 0; i < states.Count; i++)
            {
                Layer layer = Network.Layers[i];
                if (layer is RecurrentLayer)
                {
                    RecurrentState state = (RecurrentState)states[i];
                    ((RecurrentLayer)layer).FeedForward(state);
                }
                else
                {
                    FeedForwardState state = (FeedForwardState)states[i];
                    ((FeedForwardLayer)layer).FeedForward(state);
                }
            }
        }
    }
}
