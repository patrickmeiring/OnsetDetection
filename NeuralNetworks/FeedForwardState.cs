﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection.NeuralNetworks
{
    class FeedForwardState
    {
        public FeedForwardState(double[] inputs, double[] outputs)
        {
            this.Inputs = inputs;
            this.Outputs = outputs;
            this.WeightedSums = new double[outputs.Length];
        }

        public readonly double[] Inputs;
        public readonly double[] WeightedSums;
        public readonly double[] Outputs;
    }
}
