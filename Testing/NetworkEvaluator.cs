using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OnsetDetection.NeuralNetworks;
using System.IO;

namespace OnsetDetection.Testing
{
    class NetworkEvaluator
    {
        private Dataset dataset;
        public NetworkEvaluator()
        {
            dataset = Dataset.Load();
        }

        public bool LogToConsole { get; set; }

        public void Evaluate(NetworkConfiguration configuration)
        {
            RecurrentNetwork network = new RecurrentNetwork(configuration, 144, 20, 20, 20);
            RecurrentNetworkTrainer trainer = new RecurrentNetworkTrainer(network);

            trainer.LearningCoefficient = configuration.LearningCoefficient;
            trainer.NoDetectionValue = configuration.NoDetectionValue;
            trainer.DetectionValue = configuration.DetectionValue;
            trainer.Momentum = configuration.Momentum;

            string path = Path.Combine(Paths.LogsDirectory, configuration.Name + ".csv");
            StreamWriter writer = new StreamWriter(path);
            configuration.WriteConfiguration(writer);

            writer.WriteLine("Time,Epoch,Mean Abs. Weight,Train F-Score,Train Recall,Train Precision,Validation F-Score,Validation Recall,Validation Precision,Test F-Score,Test Recall, Test Precision");

            NetworkScorer trainingScorer = new NetworkScorer();
            NetworkScorer validationScorer = new NetworkScorer();
            NetworkScorer testingScorer = new NetworkScorer();

            try
            {
                for (int i = 1; i < configuration.Epochs; i++)
                {
                    foreach (TrainingSample sample in dataset.TrainingSamples)
                    {
                        trainer.Train(sample, trainingScorer);
                    }
                    foreach (TrainingSample sample in dataset.ValidationSamples)
                    {
                        trainer.FeedForward(sample, validationScorer);
                    }
                    foreach (TrainingSample sample in dataset.TestingSamples)
                    {
                        trainer.FeedForward(sample, testingScorer);
                    }
                    writer.WriteLine("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}", DateTime.Now, i, network.MeanAbsoluteWeight, 
                        trainingScorer.FScore, trainingScorer.Recall, trainingScorer.Precision,
                        validationScorer.FScore, validationScorer.Recall, validationScorer.Precision,
                        testingScorer.FScore, testingScorer.Recall, testingScorer.Precision
                        );
                    writer.Flush();

                    if (LogToConsole)
                    {
                        Console.WriteLine("{0:D3}    F-Score:{1:0.0000}   Recall:{2:0.0000}   Precision:{3:0.0000}   Mean Abs. Weight:{4:0.0000}", i, trainingScorer.FScore, trainingScorer.Recall, trainingScorer.Precision, network.MeanAbsoluteWeight);
                    }
                    trainingScorer.ResetScores();
                    validationScorer.ResetScores();
                    testingScorer.ResetScores();
                }
            }
            catch (Exception ex)
            {
                writer.WriteLine("{0},Aborted {1} at {2}", DateTime.Now, ex.Message, ex.StackTrace);
            }
            //Console.ReadLine();
        }
    }
}
