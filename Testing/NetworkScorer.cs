using OnsetDetection.NeuralNetworks;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection.Testing
{
    class NetworkScorer
    {
        public NetworkScorer()
        {
            ResetScores();
        }

        private int falsePositives;
        private int truePositives;
        private int falseNegatives;

        private const double MatchingTolerance = 0.025;

        public void Score(List<NetworkState> states, TrainingSample sample)
        {
            var availableOnsets = sample.Frames.Where(frame => frame.IsOnset).ToList();
            int totalSelection = availableOnsets.Count(); // tp + fn

            for (int i = 1; i < states.Count; i++)
            {
                NetworkState state = states[i];
                double time = sample.Frames[i - 1].Frame.Start;

                Debug.Assert(state.Output.Length == 1);
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
