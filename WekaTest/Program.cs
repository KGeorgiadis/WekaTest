using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ikvm.runtime;
using weka;


namespace WekaTest
{
    class Program
    {

        public static void Main(string[] args)
        {
            Console.WriteLine("Hello Java, from C#!");
            //classifyTest();
            BayesTest();
            Console.ReadKey();
        }

        const int percentSplit = 66;

        public static void BayesTest()
        {
            try
            {
                weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader("iris.arff"));
                insts.setClassIndex(insts.numAttributes() - 1);

                //weka.classifiers.Classifier cl = new weka.classifiers.trees.J48();
                weka.classifiers.Classifier cl = new weka.classifiers.bayes.BayesNet();
                Console.WriteLine("Performing " + percentSplit + "% split evaluation.");

                //randomize the order of the instances in the dataset.
                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);
                weka.core.Instances test = new weka.core.Instances(insts, 0, 0);


                cl.buildClassifier(train);
                //print model
                Console.WriteLine(cl);

                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = cl.classifyInstance(currentInst);
                    test.add(currentInst);
                    
                    double[] prediction = cl.distributionForInstance(currentInst);

                    for (int x = 0; x < prediction.Length; x++)
                    {
                        Console.WriteLine("Probability of class [{0}] for [{1}] is: {2}", currentInst.classAttribute().value(x), currentInst, Math.Round(prediction[x],4));
                    }
                    Console.WriteLine();
                    
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                Console.WriteLine(numCorrect + " out of " + testSize + " correct (" +
                            (double)((double)numCorrect / (double)testSize * 100.0) + "%)");

                // Train the model
                weka.classifiers.Evaluation eTrain = new weka.classifiers.Evaluation(train);
                eTrain.evaluateModel(cl, train);

                // Print the results as in Weka explorer:
                //Print statistics
                String strSummaryTrain = eTrain.toSummaryString();
                Console.WriteLine(strSummaryTrain);

                //Print detailed class statistics
                Console.WriteLine(eTrain.toClassDetailsString());

                //Print confusion matrix
                Console.WriteLine(eTrain.toMatrixString());

                // Get the confusion matrix
                double[][] cmMatrixTrain = eTrain.confusionMatrix();


                // Test the model
                weka.classifiers.Evaluation eTest = new weka.classifiers.Evaluation(test);
                eTest.evaluateModel(cl, test);

                // Print the results as in Weka explorer:
                //Print statistics
                String strSummaryTest = eTest.toSummaryString();
                Console.WriteLine(strSummaryTest);

                //Print detailed class statistics
                Console.WriteLine(eTest.toClassDetailsString());

                //Print confusion matrix
                Console.WriteLine(eTest.toMatrixString());

                // Get the confusion matrix
                double[][] cmMatrixTest = eTest.confusionMatrix();

            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
            }

        }
    }
}
