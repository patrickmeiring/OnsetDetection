using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection
{
    class RecurrentNetwork
    {
        private Layer[] layers;

        /// <summary>
        /// Creates a Recurrent Neural Network with the specified number of layers and one output Neuron.
        /// </summary>
        /// <param name="layerSize"></param>
        public RecurrentNetwork(params int[] layerSize)
        {
            layers = new Layer[layerSize.Length];
            for (int i = 0; i < layers.Length - 1; i++)
            {
                //layers[i] = new FeedForwardLayer(layerSize[i + 1], layerSize[i]);
                layers[i] = new RecurrentLayer(layerSize[i + 1], layerSize[i]);
            }
            layers[layers.Length - 1] = new FeedForwardLayer(2, layerSize[layerSize.Length - 1]);
        }

        public IList<Layer> Layers
        {
            get { return layers; }
        }
    }
}
