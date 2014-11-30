using OnsetDetection.Diagnostics;
using OnsetDetection.NeuralNetworks;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using OnsetDetection.Testing;
namespace OnsetDetection
{
    class Program
    {
        static Queue<NetworkConfiguration> tests;

        static void Main(string[] args)
        {
            NetworkConfiguration configuration = new NetworkConfiguration()
            {
                DetectionValue = 1.0,
                Epochs = 100000,
                LearningCoefficient = 0.00001,
                Momentum = 0.9,
                Name = "",
                NoDetectionValue = 0.0,
                Seed = 0,
                WeightInitialisationMethod = RandomType.Linear,
                WeightInitialisationSize = 1.0
            };

            NetworkEvaluator evaluator = new NetworkEvaluator();
            evaluator.Evaluate(configuration);
            //tests = new Queue<NetworkConfiguration>(NetworkConfiguration.Read());

            //new Thread(TestConfiguration).Start();
            //new Thread(TestConfiguration).Start();
            //new Thread(TestConfiguration).Start();
            //new Thread(TestConfiguration).Start();


            Console.ReadLine();
        }

        static void TestConfiguration()
        {
            NetworkEvaluator evaluator = new NetworkEvaluator();
            while (true)
            {
                NetworkConfiguration configuration;
                lock (tests)
                {
                    if (tests.Count == 0) break;
                    configuration = tests.Dequeue();
                }
                Console.WriteLine("[{0}] Starting test of {1}", DateTime.Now, configuration.Name);
                evaluator.Evaluate(configuration);
            }
        }
    }
}
