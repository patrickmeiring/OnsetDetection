﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection
{
    class FeedForwardTrainingState
    {
        public FeedForwardTrainingState(double[] inputErrors, double[] errors)
        {
            this.InputErrors = inputErrors;
            this.Errors = errors;
            this.BiasWeightErrors = new double[errors.Length];
            this.InputWeightErrors = new double[inputErrors.Length, errors.Length];
        }

        public FeedForwardState Now { get; set; }

        /// <summary>
        /// The partial derivative of the so-called "Objective" or "Error" function with respect to 
        /// the input of the output unit.
        /// </summary>
        public readonly double[] Errors;

        /// <summary>
        /// The partial derivative of the so-called "Objective" or "Error" function with respect to
        /// the input of the input unit.
        /// </summary>
        public readonly double[] InputErrors;


        public readonly double[] BiasWeightErrors;
        
        /// <summary>
        /// The partial derivative of the so-called "Objective" or "Error" function with respect to 
        /// each of the weights between the input and this layer.
        /// </summary>
        public readonly double[,] InputWeightErrors;
    }
}