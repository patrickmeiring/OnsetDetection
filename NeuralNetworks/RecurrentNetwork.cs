using OnsetDetection.Testing;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection.NeuralNetworks
{
    class RecurrentNetwork
    {
        private Layer[] layers;
        public readonly NetworkConfiguration Configuration;

        /// <summary>
        /// Creates a Recurrent Neural Network with the specified number of layers and one output Neuron.
        /// </summary>
        /// <param name="layerSize"></param>
        public RecurrentNetwork(NetworkConfiguration configuration, params int[] layerSize)
        {
            this.Configuration = configuration;
            configuration.InitialiseRandom();

            layers = new Layer[layerSize.Length];
            for (int i = 0; i < layers.Length - 1; i++)
            {
                //layers[i] = new FeedForwardLayer(layerSize[i + 1], layerSize[i]);
                layers[i] = new RecurrentLayer(configuration, layerSize[i + 1], layerSize[i]);
            }
            layers[layers.Length - 1] = new FeedForwardLayer(configuration, 1, layerSize[layerSize.Length - 1]);
        }

        public IList<Layer> Layers
        {
            get { return layers; }
        }

        public double MeanAbsoluteWeight
        {
            get
            {
                return layers.Sum(l => l.SumAbsoluteWeight) / layers.Sum(l => l.WeightCount);
            }
        }
    }
}
