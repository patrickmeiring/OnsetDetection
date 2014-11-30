using OnsetDetection.Testing;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection.NeuralNetworks
{
    class RecurrentNetworkTrainer
    {
        public readonly RecurrentNetwork Network;

        public double LearningCoefficient { get; set; }
        public double Momentum { get; set; }
        public double DetectionValue { get; set; }
        public double NoDetectionValue { get; set; }

        public RecurrentNetworkTrainer(RecurrentNetwork network)
        {
            this.Network = network;
            this.trainingState = new NetworkTrainingState(network);
        }

        private NetworkConfiguration Configuration
        {
            get { return Network.Configuration; }
        }

        private NetworkTrainingState trainingState; 
        
        
        public void Train(TrainingSample sample, NetworkScorer scorer)
        {
            List<NetworkState> states = FeedForward(sample, scorer);

            trainingState.Prepare(momentum: Momentum);

            // Backpropogate through time.
            for (int i = states.Count - 1; i >= 1; i--)
            {
                // Get the corresponding frame.
                TrainingFrame frame = sample.Frames[i - 1];
                NetworkState current = states[i];
                NetworkState last = states[i - 1];

                //bool annotatedOnsetNearby = IsAnnotatedOnsetNearby(sample, states, i);

                double onCorrect = frame.IsOnset ? DetectionValue : NoDetectionValue;
                //double offCorrect = frame.IsOnset ? 0.0 : 1.0;

                trainingState.Errors[0] = (current.Output[0] - onCorrect);
               // trainingState.Errors[1] = (current.Output[1] - offCorrect);

                trainingState.BackPropogate(last: last, now: current);
            }

            // Apply the required weight changes.
            trainingState.ApplyWeightChanges(LearningCoefficient);
            
        }

        public List<NetworkState> FeedForward(TrainingSample sample, NetworkScorer scorer)
        {
            List<NetworkState> states = new List<NetworkState>();

            // Initialise with a zero before-starting state.
            NetworkState state = new NetworkState(Network);
            states.Add(state);

            // Forward propogate through time
            foreach (TrainingFrame trainingFrame in sample.Frames)
            {
                state = new NetworkState(Network, state);
                Array.Copy(trainingFrame.Frame.Values, state.Input, state.Input.Length);
                state.FeedForward();
                states.Add(state);
            }

            scorer.Score(states, sample);
            return states;
        }

        //private bool IsAnnotatedOnsetNearby(TrainingSample sample, List<NetworkState> states, int index)
        //{
        //    bool annotatedOnsetNearby = false;
        //    for (int i = -2; i < 2; i++)
        //    {
        //        int stateIndex = index + i;
        //        if (stateIndex <= 0 || stateIndex >= states.Count) continue;

        //        // The corresponding frame is at [stateIndex - 1] since we have an extra 'zero' state at the start, for which there is no frame.
        //        bool annotatedOnset = sample.Frames[stateIndex - 1].IsOnset;
        //        annotatedOnsetNearby |= annotatedOnset;
        //    }
        //    return annotatedOnsetNearby;
        //}
     
    }
}
