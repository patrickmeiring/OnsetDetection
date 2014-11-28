using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection
{
    public class PreprocessingWindow
    {
        private static readonly int[] barkBands = 
        {
           20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500 
        };

        private readonly double[] realBuffer;
        private readonly double[] imaginaryBuffer;
        private readonly double[][] outputHistory;
        private int position;

        private readonly int bands;
        private readonly int samplingFrequency;

        public readonly int Step;

        public PreprocessingWindow(int windowSize, int samplingFrequency, int step)
        {
            realBuffer = new double[windowSize];
            imaginaryBuffer = new double[windowSize];

            this.samplingFrequency = samplingFrequency;
            this.Step = step;
            this.bands = Math.Min(barkBands.Select(band => band < (samplingFrequency / 2)).Count(), barkBands.Length - 1);

            int historyLength = Math.Max((int)Math.Round((windowSize / step) / 2.0), 1) + 1;
            this.outputHistory = new double[historyLength][];
            for (int i = 0; i < historyLength; i++)
            {
                double[] history = new double[bands];
                for (int b = 0; b < bands; b++)
                {
                    history[b] = Math.Log(1.0);
                }
                outputHistory[i] = history;
            }

            // We need to fill up the history before we can provide our first output.
            this.position = 0;// -outputHistory.Length;
        }

        public void Process(double[] circularList, int lastIndex)
        {
            Debug.Assert(circularList.Length >= realBuffer.Length);
            int firstIndex = (circularList.Length + (lastIndex - realBuffer.Length)) % circularList.Length;

            // Fill realBuffer with the next set of samples.
            int elementsToCopyFirst = Math.Min(realBuffer.Length, circularList.Length - firstIndex);
            Array.Copy(circularList, firstIndex, realBuffer, 0, elementsToCopyFirst);
            Array.Copy(circularList, 0, realBuffer, elementsToCopyFirst, realBuffer.Length - elementsToCopyFirst);

            // Perform a discrete fourier transform on the input buffer.
            DiscreteFourierTransform.Transform(realBuffer, imaginaryBuffer);
            
            // Take the absolute values of the real and imaginary frequency components
            // and store it in the realBuffer.
            DiscreteFourierTransform.KeepRealComponent(realBuffer, imaginaryBuffer);

            int historyPosition = GetHistoryIndex(position);
            ApplyBarkFilterbank(realBuffer, outputHistory[historyPosition]);

            position++;

        }

        public int OutputSize
        {
            get { return bands; }
        }

        //public bool IsOutputAvailable
        //{
        //    get { return position >= 0; }
        //}


        public void PrepareAbsoluteFrame(double[] destination, int index)
        {
            double[] current = outputHistory[GetHistoryIndex(position - 1)];
            Array.Copy(current, 0, destination, index, current.Length);
        }

        public void PrepareDifferenceFrame(double[] destination, int index)
        {
            double[] current = outputHistory[GetHistoryIndex(position - 1)];
            double[] last = outputHistory[GetHistoryIndex(position)];
            Debug.Assert(current.Length == last.Length);
            for (int i = 0; i < current.Length; i++)
            {
                double difference = current[i] - last[i];
                destination[index + i] = Math.Max(0.0, difference);
            }
        }

        private int GetHistoryIndex(int position)
        {
            return (outputHistory.Length + position) % outputHistory.Length;
        }

        private void ApplyBarkFilterbank(double[] input, double[] output)
        {
            Debug.Assert(input.Length == realBuffer.Length);
            Debug.Assert(output.Length == bands);

            double frequencyIncrement = (samplingFrequency / (input.Length - 1.0));

            for (int b = 0; b < bands; b++)
            {
                double lowerFreq = barkBands[b];
                double upperFreq = Math.Min(barkBands[b + 1], samplingFrequency / 2);

                double lowerPosition = lowerFreq / frequencyIncrement;
                double centerPosition = ((lowerFreq + upperFreq) / 2.0) / frequencyIncrement;
                double upperPosition = upperFreq / frequencyIncrement;

                double height = 2.0 / (upperPosition - lowerPosition);

                double sum = 0.0;
                for (int i = (int)Math.Ceiling(lowerPosition); i <= (int)Math.Floor(upperPosition); i++)
                {
                    double position = i;
                    if (position < centerPosition)
                    {
                        sum += input[i] * ((position - lowerPosition) / (centerPosition - lowerPosition));// *height;
                    }
                    else
                    {
                        sum += input[i] * ((upperPosition - position) / (upperPosition - centerPosition));// *height;
                    }
                }
                output[b] = Math.Log(sum + 1);
            }
        }
    }
}
