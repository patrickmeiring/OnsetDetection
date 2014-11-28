using OnsetDetection;
using OnsetDetection.Datasets;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection
{
    class OnsetRealignment
    {
        private const int AccuracyInSamples = 32;
        private const int FourierTransformSize = 512;

        public static IEnumerable<double> RealignOnsets(TrackInfo track)
        {
            try
            {
                Console.Write("Realigning Onsets for {0}... ", track.TrackName);
                double[] energies = CalculateEnergies(track);
                var onsets = track.GetOriginalOnsets().Select(time => time * 44100);

                int optimalCorrection = int.MinValue;
                double optimalFitness = double.MinValue;

                int minimumAdjustment = (int)Math.Max(-onsets.Min() - FourierTransformSize / 2, -4410);
                for (int i = minimumAdjustment; i < 4410; i++)
                {
                    double fitness = EvaluateFitness(energies, onsets, i);
                    if (fitness > optimalFitness)
                    {
                        optimalFitness = fitness;
                        optimalCorrection = i;
                    }
                }
                Console.WriteLine("Done.");
                return onsets.Select(onset => (onset + optimalCorrection) / 44100);
            }
            catch (Exception ex)
            { throw;
            }
        }

        private static double EvaluateFitness(double[] energies, IEnumerable<double> samples, int samplesOffset)
        {
            double sum = 0.0;
            foreach (double onset in samples)
            {
                int position = (int)Math.Round((onset + samplesOffset + (FourierTransformSize / 2.0)) / AccuracyInSamples);
                sum += energies[position];
            }
            return sum;
        }

        private static double[] CalculateEnergies(TrackInfo trackInfo)
        {
            using (WaveFile wave = trackInfo.GetWaveFile())
            {
                wave.Open();
                Debug.Assert(wave.SampleRate == 44100);
                PreprocessingWindow window = new PreprocessingWindow(512, 44100, AccuracyInSamples);
                double[] samples = new double[wave.Length + 512];
                double[] result = new double[wave.Length / AccuracyInSamples];
                Debug.Assert(wave.Read(samples) == wave.Length);

                double[] bands = new double[window.OutputSize];

                for (int i = 0; i < result.Length; i++)
                {
                    window.Process(samples, i * AccuracyInSamples);
                    window.PrepareDifferenceFrame(bands, 0);
                    result[i] = bands.Sum();
                }
                return result;
            }

        }
    }
}
