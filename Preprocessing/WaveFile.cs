using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection
{
    public class WaveFile : IDisposable
    {
        private readonly string path;
        private BinaryReader reader;
        private int sampleRate;
        private int position;
        private int length;
        private int channels;

        public WaveFile(string path)
        {
            this.path = path;
            this.reader = null;
        }

        public void Open()
        {
            FileStream stream = new FileStream(path, FileMode.Open);
            reader = new BinaryReader(stream, Encoding.ASCII);
            ReadHeader();
        }

        private void ReadHeader()
        {
            // ChunkID
            string chunkID = new string(reader.ReadChars(4));
            Debug.Assert(chunkID == "RIFF");

            // ChunkSize
            reader.ReadInt32();

            // Format
            string format = new string(reader.ReadChars(4));
            Debug.Assert(format == "WAVE");

            // Subchunk1ID
            string subchunk1Id = new string(reader.ReadChars(4));
            Debug.Assert(subchunk1Id == "fmt ");

            int subchunk1Size = reader.ReadInt32();
            Debug.Assert(subchunk1Size == 16, "The file must be PCM encoded"); // PCM encoded.

            int audioFormat = reader.ReadUInt16();
            Debug.Assert(audioFormat == 1, "The file must be PCM encoded"); // PCM.

            channels = reader.ReadUInt16();
            Debug.Assert(channels <= 2, "There must be no more than two channels"); // Mono or Stereo only

            sampleRate = reader.ReadInt32();
            Debug.Assert(sampleRate == 44100, "Sample rate must be 44100");//>= 8000, "Sample rate must be above 8000"); // Sample rate

            int byteRate = reader.ReadInt32();
            int blockAlign = reader.ReadUInt16();
            int bitsPerSample = reader.ReadUInt16();
            Debug.Assert(bitsPerSample == 16); // 16 bits per sample.

            string subchunk2Id = new string(reader.ReadChars(4));
            Debug.Assert(subchunk2Id == "data");

            int subchunk2Size = reader.ReadInt32();
            length = subchunk2Size / (channels * bitsPerSample / 8);
        }

        public int SampleRate
        {
            get { return sampleRate; }
        }

        public int Length
        {
            get { return length; }
        }

        public int Channels
        {
            get { return channels; }
        }

        public int Read(double[] buffer)
        {
            int samplesToRead = Math.Min(buffer.Length, length - position);
            for (int i = 0; i < samplesToRead; i++)
            {
                buffer[i] = ReadSample();
            }
            position += samplesToRead;
            return samplesToRead;
        }

        private double ReadSample()
        {
            if (channels == 1)
            {
                return reader.ReadInt16() / 32767.0;
            }
            else if (channels == 2)
            {
                double leftSample = reader.ReadInt16() / 32767.0;
                double rightSample = reader.ReadInt16() / 32767.0;
                return (leftSample + rightSample) / 2.0;
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        public void Dispose()
        {
            if (reader != null)
            {
                reader.Dispose();
                reader = null;
            }
        }
    }
}
