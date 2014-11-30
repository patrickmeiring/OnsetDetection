using OnsetDetection;
using OnsetDetection.Datasets;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection
{
    class OnsetRealignment
    {
        private const int AccuracyInSamples = 32;
        private const int FourierTransformSize = 512;

        private static StreamWriter writer;
        private static void Log(TrackInfo track, int correction)
        {
            if (writer == null)
            {
                bool exists = File.Exists(Paths.AdjustedOnsetsLog);
                writer = new StreamWriter(Paths.AdjustedOnsetsLog);
                if (!exists)
                {
                    writer.WriteLine("Full Track Name,Author,Track Name,Start Time,End Time,Adjustment (samples), Adjustment (seconds)");
                }
            }
            writer.WriteLine("{0},{1},{2},{3},{4},{5},{6}", track.FullName, track.AuthorInitials, track.TrackName, track.StartTime, track.EndTime, correction, ((double)correction) / 44100.0);
            writer.Flush();
        }


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
                //int maximumAdjustment = (int)Math.Min(onsets.Max() - FourierTransformSize / 2, 4410);
                for (int i = minimumAdjustment; i < 4410; i++)
                {
                    try
                    {
                        double fitness = EvaluateFitness(energies, onsets, i);
                        if (fitness > optimalFitness)
                        {
                            optimalFitness = fitness;
                            optimalCorrection = i;
                        }
                    }
                    catch (IndexOutOfRangeException) { }
                }
                Console.WriteLine("Done with correction {0}.", optimalCorrection);

                Log(track, optimalCorrection);

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
