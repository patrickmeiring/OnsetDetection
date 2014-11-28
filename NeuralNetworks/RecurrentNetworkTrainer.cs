using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection
{
    class RecurrentNetworkTrainer
    {
        public readonly RecurrentNetwork Network;

        public double LearningCoefficient { get; set; }

        public RecurrentNetworkTrainer(RecurrentNetwork network)
        {
            this.Network = network;
            ResetScores();
        }

        private int falsePositives;
        private int truePositives;
        private int falseNegatives;
        
        public void Train(TrainingSample sample)
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

            Score(states, sample);

            NetworkTrainingState trainingState = new NetworkTrainingState(Network);
            
            // Backpropogate through time.
            for (int i = states.Count - 1; i >= 1; i--)
            {
                // Get the corresponding frame.
                TrainingFrame frame = sample.Frames[i - 1];
                NetworkState current = states[i];
                NetworkState last = states[i - 1];

                double onCorrect = frame.IsOnset ? 1.0 : 0.0;
                double offCorrect = frame.IsOnset ? 0.0 : 1.0;

                trainingState.Errors[0] = (current.Output[0] - onCorrect);
                trainingState.Errors[1] = (current.Output[1] - offCorrect);
               
                trainingState.BackPropogate(last: last, now: current);
            }

            // Apply the required weight changes.
            trainingState.ApplyWeightChanges(LearningCoefficient);
        }

        private const double MatchingTolerance = 0.025;

        private void Score(List<NetworkState> states, TrainingSample sample)
        {
            var availableOnsets = sample.Frames.Where(frame => frame.IsOnset).ToList();
            int totalSelection = availableOnsets.Count(); // tp + fn
            
            for (int i = 1; i < states.Count; i++)
            {
                NetworkState state = states[i];
                double time = sample.Frames[i - 1].Frame.Start;

                Debug.Assert(state.Output.Length == 2);
                if (state.Output[0] < 0.25) continue;// < state.Output[1]) continue;

                TrainingFrame matchedOnset = availableOnsets.FirstOrDefault(onset => Math.Abs(onset.Frame.Start - time) < MatchingTolerance);
                if (matchedOnset != null)
                {
                    availableOnsets.Remove(matchedOnset);
                    truePositives++;
                }
                else
                {
                    falsePositives++;
                }
            }
            falseNegatives += availableOnsets.Count;
        }

        public void ResetScores()
        {
            truePositives = 0;
            falseNegatives = 0;
            falsePositives = 0;
        }

        public double Precision
        {
            get { return ((double)truePositives) / (truePositives + falsePositives); }
        }

        public double Recall
        {
            get { return ((double)truePositives) / (truePositives + falseNegatives); }
        }

        public double FScore
        {
            get { return 2.0 * (Precision * Recall) / (Precision + Recall); }
        }
    }
}
