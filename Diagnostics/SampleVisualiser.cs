using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection.Diagnostics
{
    public class SampleVisualiser
    {
        public static Bitmap Draw(TrainingSample sample)
        {
            var frames = sample.Frames;
            int height = frames[0].Frame.Values.Length;
            Debug.Assert(height % 6 == 0);

            int bands = height / 6;
            Bitmap result = new Bitmap(frames.Length, bands * 6 + 5);
            
            for (int x = 0; x < frames.Length; x++)
            {
                TrainingFrame frame = frames[x];
                DrawRow(result, x, frame);
            }
            return result;
        }

        private static void DrawRow(Bitmap bitmap, int x, TrainingFrame frame)
        {
            double[] values = frame.Frame.Values;
            int bands = (bitmap.Height - 5) / 6;
            int y = 0;
            int index = 0;
            for (int part = 0; part < 6; part++)
            {
                for (int i = 0; i < bands; i++)
                {
                    bitmap.SetPixel(x, y++, GetColor(values[index++], frame.IsOnset));
                }

                if (part < 5)
                {
                    bitmap.SetPixel(x, y++, Color.Black);
                }
            }
        }

        private static Color GetColor(double value, bool isOnset)
        {
            double brightness = isOnset ? 255.999 : 127;
            value /= 1.5;
            value = Math.Min(2.0, value);
            double red, green, blue;
            if (value < 1.0)
            {
                blue = gammaTransform(1.0 - value);
                green = gammaTransform(value);
                double max = Math.Max(blue, green);
                blue /= max;
                green /= max;
                red = 0;
            }
            else
            {
                blue = 0;
                green = gammaTransform(2.0 - value);
                red = gammaTransform(value - 1.0);
                double max = Math.Max(green, red);
                green /= max;
                red /= max;
                blue = 0;
            }
            return Color.FromArgb((int)(red * brightness), (int)(green * brightness), (int)(blue * brightness));
        }

        private static double gammaTransform(double linear)
        {
            if (linear <= 0.0031308)
            {
                return 12.92 * linear;
            }
            else
            {
                return (1.055) * Math.Pow(linear, 1.0 / 2.4) - 0.055;
            }
        }

    }
}
