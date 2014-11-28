using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection
{
    /// <summary>
    /// Based on ONLINE REAL-TIME ONSET DETECTION WITH RECURRENT NEURAL NETWORKS
    /// </summary>
    public class Preprocessing
    {
        private const int LONG_SAMPLE = 2048;/// 1024;
        private const int MEDIUM_SAMPLE = 1024;// 512;
        private const int SHORT_SAMPLE = 512;// 256;

        private int bufferPosition;
        private int sample;

        private double[] buffer;

        public readonly int SamplingFrequency;
        public readonly int Step;

        private PreprocessingWindow shortWindow;
        private PreprocessingWindow mediumWindow;
        private PreprocessingWindow longWindow;

        public Preprocessing(int samplingFrequency, int step)
        {
            shortWindow = new PreprocessingWindow(SHORT_SAMPLE, samplingFrequency, step);
            mediumWindow = new PreprocessingWindow(MEDIUM_SAMPLE, samplingFrequency, step);
            longWindow = new PreprocessingWindow(LONG_SAMPLE, samplingFrequency, step);

            this.Step = step;
            this.SamplingFrequency = samplingFrequency;
            sample = 0;
            buffer = new double[LONG_SAMPLE];
            bufferPosition = 0;
        }

        public IList<Frame> Input(WaveFile file)
        {
            int length;
            double[] sampleBuffer = new double[2048];
            List<Frame> samples = new List<Frame>();
            do
            {
                length = file.Read(sampleBuffer);
                samples.AddRange(Input(sampleBuffer, length));
            }
            while (length == sampleBuffer.Length);
            return samples;
        }

        public IEnumerable<Frame> Input(double[] samples, int length)
        {
            int index = 0;
            while (InputUntilStep(samples, length, ref index))
            {
                Debug.Assert(sample % Step == 0);
                shortWindow.Process(buffer, bufferPosition);
                mediumWindow.Process(buffer, bufferPosition);
                longWindow.Process(buffer, bufferPosition);

                Debug.Assert(shortWindow.OutputSize == mediumWindow.OutputSize && mediumWindow.OutputSize == longWindow.OutputSize);
                double[] frame = new double[shortWindow.OutputSize * 2 + mediumWindow.OutputSize * 2 + longWindow.OutputSize * 2];
                shortWindow.PrepareAbsoluteFrame(frame, 0);
                shortWindow.PrepareDifferenceFrame(frame, shortWindow.OutputSize);
                mediumWindow.PrepareAbsoluteFrame(frame, shortWindow.OutputSize * 2);
                mediumWindow.PrepareDifferenceFrame(frame, shortWindow.OutputSize * 2 + mediumWindow.OutputSize);
                longWindow.PrepareAbsoluteFrame(frame, shortWindow.OutputSize * 2 + mediumWindow.OutputSize * 2);
                longWindow.PrepareDifferenceFrame(frame, shortWindow.OutputSize * 2 + mediumWindow.OutputSize * 2 + longWindow.OutputSize);

                double startTime = (sample * 1.0 - Step) / SamplingFrequency;
                double endTime = (sample * 1.0) / SamplingFrequency;
                yield return new Frame(startTime, endTime, frame);
            }
        }

        private bool InputUntilStep(double[] samples, int samplesLength, ref int index)
        {
            int samplesUntilStep = Step - (sample % Step);
            int samplesToCopy = Math.Min(samplesUntilStep, samples.Length - index);

            int firstPass = Math.Min(samplesToCopy, buffer.Length - bufferPosition);
            Array.Copy(samples, index, buffer, bufferPosition, firstPass);
            Array.Copy(samples, index + firstPass, buffer, 0, samplesToCopy - firstPass);

            bufferPosition = (bufferPosition + samplesToCopy) % buffer.Length;
            sample += samplesToCopy;
            index += samplesToCopy;
            return samplesToCopy == samplesUntilStep;
        }
    }
}
