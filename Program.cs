using OnsetDetection.Diagnostics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnsetDetection
{
    class Program
    {
        static void Main(string[] args)
        {

            RecurrentNetwork network = new RecurrentNetwork(144, 20, 20, 20);//, 20, 20, 20);//, 20, 20, 20);
            Dataset dataset = Dataset.Load();

            RecurrentNetworkTrainer trainer = new RecurrentNetworkTrainer(network);

            trainer.LearningCoefficient = 0.0001;
            for (int i = 1; ; i++)
            {
                foreach (TrainingSample sample in dataset.Samples)
                {
                    trainer.Train(sample);
                }


                trainer.LearningCoefficient *= 0.9;
                Console.WriteLine("{0}    F-Score:{1:0.0000}   Recall:{2:0.0000}   Precision:{3:0.0000}", i, trainer.FScore, trainer.Recall, trainer.Precision);
                trainer.ResetScores();
            } Console.ReadLine();
        }
    }
}
