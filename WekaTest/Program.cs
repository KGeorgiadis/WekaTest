using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ikvm.runtime;
using weka;
using weka.clusterers;
using weka.core.converters;
using java.io;
using weka.core;
using weka.classifiers.meta;
using weka.attributeSelection;
using weka.classifiers.evaluation;

namespace WekaTest
{
    class Program
    {

        public static void Main(string[] args)
        {
            //BayesTest();
            //DensityBasedClusterer();
            //CreateArffFiles();
            //cvdTest();
            Test();
            System.Console.ReadKey();
        }

        const int percentSplit = 66;

        public static void BayesTest()
        {
            try
            {
                weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader("iris.arff"));
                insts.setClassIndex(insts.numAttributes() - 1);

                weka.classifiers.Classifier cl = new weka.classifiers.bayes.BayesNet();
                System.Console.WriteLine("Performing " + percentSplit + "% split evaluation.");

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
                System.Console.WriteLine(cl);

                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = cl.classifyInstance(currentInst);
                    test.add(currentInst);

                    double[] prediction = cl.distributionForInstance(currentInst);

                    for (int x = 0; x < prediction.Length; x++)
                    {
                        System.Console.WriteLine("Probability of class [{0}] for [{1}] is: {2}", currentInst.classAttribute().value(x), currentInst, Math.Round(prediction[x], 4));
                    }
                    System.Console.WriteLine();

                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                System.Console.WriteLine(numCorrect + " out of " + testSize + " correct (" +
                            (double)((double)numCorrect / (double)testSize * 100.0) + "%)");

                // Train the model
                weka.classifiers.Evaluation eTrain = new weka.classifiers.Evaluation(train);
                eTrain.evaluateModel(cl, train);

                // Print the results as in Weka explorer:
                //Print statistics
                String strSummaryTrain = eTrain.toSummaryString();
                System.Console.WriteLine(strSummaryTrain);

                //Print detailed class statistics
                System.Console.WriteLine(eTrain.toClassDetailsString());

                //Print confusion matrix
                System.Console.WriteLine(eTrain.toMatrixString());

                // Get the confusion matrix
                double[][] cmMatrixTrain = eTrain.confusionMatrix();


                // Test the model
                weka.classifiers.Evaluation eTest = new weka.classifiers.Evaluation(test);
                eTest.evaluateModel(cl, test);

                // Print the results as in Weka explorer:
                //Print statistics
                String strSummaryTest = eTest.toSummaryString();
                System.Console.WriteLine(strSummaryTest);

                //Print detailed class statistics
                System.Console.WriteLine(eTest.toClassDetailsString());

                //Print confusion matrix
                System.Console.WriteLine(eTest.toMatrixString());

                // Get the confusion matrix
                double[][] cmMatrixTest = eTest.confusionMatrix();

            }

            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
            }

        }

        public static void DensityBasedClusterer()
        {
            try
            {
                Instances data = new Instances(new java.io.FileReader("politeness.arff"));

                MakeDensityBasedClusterer clusterer = new MakeDensityBasedClusterer();

                // set further options for EM, if necessary...                
                clusterer.setNumClusters(3);
                clusterer.buildClusterer(data);

                ClusterEvaluation eval = new ClusterEvaluation();
                eval.setClusterer(clusterer);
                eval.evaluateClusterer(data);


                /** Print Prior probabilities for each cluster
                *double[] Priors = clusterer.clusterPriors();
                *
                *for(int x=0; x<Priors.Length; x++)
                *{
                *   System.Console.WriteLine(Priors[x]);
                *}
                **/

                /**Print default capabilities of the clusterer (i.e., of the wrapper clusterer).
                *Capabilities Capa = clusterer.getCapabilities();
                *System.Console.WriteLine(Capa);
                **/

                /**Print the current settings of the clusterer.
                *String[] Opts = clusterer.getOptions();
                *for (int x = 0; x < Opts.Length; x++)
                *{
                *    System.Console.WriteLine(Opts[x]);
                *}
                **/

                //string gInfo = clusterer.globalInfo();
                //System.Console.WriteLine(gInfo);

                java.util.Enumeration enumOpts = clusterer.listOptions();
                System.Console.WriteLine(enumOpts);

                //Print all results for clusterer as in Weka
                //System.Console.WriteLine(eval.clusterResultsToString());


            }

            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
            }
        }

        public static void CreateArffFiles()
        {
            java.util.ArrayList atts;
            java.util.ArrayList attsRel;
            java.util.ArrayList attVals;
            java.util.ArrayList attValsRel;
            Instances data;
            Instances dataRel;
            double[] vals;
            double[] valsRel;
            int i;

            // 1. set up attributes
            atts = new java.util.ArrayList();
            // - numeric
            atts.Add(new weka.core.Attribute("att1"));
            // - nominal
            attVals = new java.util.ArrayList();
            for (i = 0; i < 5; i++)
                attVals.add("val" + (i + 1));

            weka.core.Attribute nominal = new weka.core.Attribute("att2", attVals);
            atts.add(nominal);
            // - string
            atts.add(new weka.core.Attribute("att3", (java.util.ArrayList)null));
            // - date
            atts.add(new weka.core.Attribute("att4", "yyyy-MM-dd"));
            // - relational
            attsRel = new java.util.ArrayList();
            // -- numeric
            attsRel.add(new weka.core.Attribute("att5.1"));
            // -- nominal
            attValsRel = new java.util.ArrayList();
            for (i = 0; i < 5; i++)
                attValsRel.Add("val5." + (i + 1));
            attsRel.add(new weka.core.Attribute("att5.2", attValsRel));
            dataRel = new Instances("att5", attsRel, 0);
            atts.add(new weka.core.Attribute("att5", dataRel, 0));

            // 2. create Instances object
            data = new Instances("MyRelation", atts, 0);

            // 3. fill with data
            // first instance
            vals = new double[data.numAttributes()];
            // - numeric
            vals[0] = Math.PI;
            // - nominal
            vals[1] = attVals.indexOf("val3");
            // - string
            vals[2] = data.attribute(2).addStringValue("This is a string!");
            // - date
            vals[3] = data.attribute(3).parseDate("2001-11-09");
            // - relational
            dataRel = new Instances(data.attribute(4).relation(), 0);
            // -- first instance
            valsRel = new double[2];
            valsRel[0] = Math.PI + 1;
            valsRel[1] = attValsRel.indexOf("val5.3");
            weka.core.Instance inst = new DenseInstance(2);
            inst.setValue(1, valsRel[0]);
            inst.setValue(1, valsRel[1]);
            dataRel.add(inst);
            // -- second instance
            valsRel = new double[2];
            valsRel[0] = Math.PI + 2;
            valsRel[1] = attValsRel.indexOf("val5.2");
            dataRel.add(inst);
            vals[4] = data.attribute(4).addRelation(dataRel);
            // add
            weka.core.Instance inst2 = new DenseInstance(4);
            inst2.setValue(1, vals[0]);
            inst2.setValue(1, vals[1]);
            inst2.setValue(1, vals[2]);
            inst2.setValue(1, vals[3]);
            data.add(inst2);

            // second instance
            vals = new double[data.numAttributes()];  // important: needs NEW array!
                                                      // - numeric
            vals[0] = Math.E;
            // - nominal
            vals[1] = attVals.indexOf("val1");
            // - string
            vals[2] = data.attribute(2).addStringValue("And another one!");
            // - date
            vals[3] = data.attribute(3).parseDate("2000-12-01");
            // - relational
            dataRel = new Instances(data.attribute(4).relation(), 0);
            // -- first instance
            valsRel = new double[2];
            valsRel[0] = Math.E + 1;
            valsRel[1] = attValsRel.indexOf("val5.4");
            dataRel.add(inst);
            // -- second instance
            valsRel = new double[2];
            valsRel[0] = Math.E + 2;
            valsRel[1] = attValsRel.indexOf("val5.1");
            dataRel.add(inst);
            vals[4] = data.attribute(4).addRelation(dataRel);
            // add
            data.add(inst2);

            data.setClassIndex(data.numAttributes() - 1);

            // 4. output data
            for (int x = 0; x < data.numInstances(); x++)
            {
                weka.core.Instance ins = data.instance(x);
                System.Console.WriteLine(ins.value(x).ToString());
            }



            return;
        }

        public static void cvdTest()
        {

            weka.core.Instances data = new weka.core.Instances(new java.io.FileReader("./data/Classification/Communication.arff"));
            data.setClassIndex(data.numAttributes() - 1);

            weka.classifiers.Classifier cls = new weka.classifiers.bayes.NaiveBayes();

            //Save BayesNet results in .txt file
            using (System.IO.StreamWriter file = new System.IO.StreamWriter("./data/Classification/Communication_Report.txt"))
            {
                int runs = 1;
                int folds = 10;

                // perform cross-validation
                for (int i = 0; i < runs; i++)
                {
                    // randomize data
                    int seed = i + 1;
                    java.util.Random rand = new java.util.Random(seed);
                    weka.core.Instances randData = new weka.core.Instances(data);
                    randData.randomize(rand);
                    if (randData.classAttribute().isNominal())
                        randData.stratify(folds);

                    weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(randData);
                    for (int n = 0; n < folds; n++)
                    {
                        weka.core.Instances train = randData.trainCV(folds, n);
                        weka.core.Instances test = randData.testCV(folds, n);
                        // build and evaluate classifier
                        //weka.classifiers.Classifier clsCopy = weka.classifiers.Classifier.makeCopy(cls);
                        cls.buildClassifier(train);
                        //eval.evaluateModel(cls, test);                

                        //Print classifier analytics for all the dataset
                        file.WriteLine("EVALUATION OF TEST DATASET.");
                        // Test the model
                        weka.classifiers.Evaluation eTest = new weka.classifiers.Evaluation(test);
                        eTest.evaluateModel(cls, test);

                        // Print the results as in Weka explorer:
                        //Print statistics
                        String strSummaryTest = eTest.toSummaryString();

                        file.WriteLine(strSummaryTest);
                        file.WriteLine();

                        //Print detailed class statistics
                        file.WriteLine(eTest.toClassDetailsString());
                        file.WriteLine();

                        //Print confusion matrix
                        file.WriteLine(eTest.toMatrixString());
                        file.WriteLine();

                        // Get the confusion matrix
                        double[][] cmMatrixTest = eTest.confusionMatrix();

                        System.Console.WriteLine("Bayesian Network results saved in Communication_Report.txt file successfully.");
                    }

                    //Print classifier analytics for all the dataset
                    file.WriteLine("EVALUATION OF ALL DATASET.");

                    cls.buildClassifier(data);

                    // Train the model
                    weka.classifiers.Evaluation eAlldata = new weka.classifiers.Evaluation(data);
                    eAlldata.evaluateModel(cls, data);

                    // Print the results as in Weka explorer:
                    //Print statistics
                    String strSummaryAlldata = eAlldata.toSummaryString();
                    file.WriteLine(strSummaryAlldata);
                    file.WriteLine();

                    //Print detailed class statistics
                    file.WriteLine(eAlldata.toClassDetailsString());
                    file.WriteLine();

                    //Print confusion matrix
                    file.WriteLine(eAlldata.toMatrixString());
                    file.WriteLine("----------------");

                    //print model
                    file.WriteLine(cls);
                    file.WriteLine();

                }
            }
        }

        public static void Test()
        {

            weka.core.Instances data = new weka.core.Instances(new java.io.FileReader("./data/Classification/Communication.arff"));
            data.setClassIndex(data.numAttributes()-1);

            weka.classifiers.Classifier cls = new weka.classifiers.bayes.BayesNet();
            

            //Save BayesNet results in .txt file
            using (System.IO.StreamWriter file = new System.IO.StreamWriter("./data/Classification/Communication_Report.txt"))
            {
                file.WriteLine("Performing " + percentSplit + "% split evaluation.");

                int runs = 1;

                // perform cross-validation
                for (int i = 0; i < runs; i++)
                {
                    // randomize data
                    int seed = i + 1;
                    java.util.Random rand = new java.util.Random(seed);
                    weka.core.Instances randData = new weka.core.Instances(data);
                    randData.randomize(rand);

                    //weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(randData);

                    int trainSize = (int)Math.Round((double)data.numInstances() * percentSplit / 100);
                    int testSize = data.numInstances() - trainSize;
                    weka.core.Instances train = new weka.core.Instances(data, 0, 0);
                    weka.core.Instances test = new weka.core.Instances(data, 0, 0);
                    train.setClassIndex(train.numAttributes() - 1);
                    test.setClassIndex(test.numAttributes() - 1);

                    //Print classifier analytics for all the dataset
                    file.WriteLine("EVALUATION OF TEST DATASET.");

                    //int numCorrect = 0;
                    for (int j = 0; j < data.numInstances(); j++)
                    {

                        weka.core.Instance currentInst = randData.instance(j);

                        if (j<trainSize)
                        {
                            train.add(currentInst);
                        }

                        else
                        {
                            test.add(currentInst);
                            /*
                            double predictedClass = cls.classifyInstance(currentInst);

                            double[] prediction = cls.distributionForInstance(currentInst);

                            for (int p = 0; p < prediction.Length; p++)
                            {
                                file.WriteLine("Probability of class [{0}] for [{1}] is: {2}", currentInst.classAttribute().value(p), currentInst, Math.Round(prediction[p], 4));
                            }
                            file.WriteLine();

                            file.WriteLine();
                            if (predictedClass == data.instance(j).classValue())
                                numCorrect++;*/
                        }

                    }

                    // build and evaluate classifier
                    cls.buildClassifier(train);

                    // Test the model
                    weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(randData);
                    eval.evaluateModel(cls, test);

                    // Print the results as in Weka explorer:
                    //Print statistics
                    String strSummaryTest = eval.toSummaryString();

                    file.WriteLine(strSummaryTest);
                    file.WriteLine();

                    //Print detailed class statistics
                    file.WriteLine(eval.toClassDetailsString());
                    file.WriteLine();

                    //Print confusion matrix
                    file.WriteLine(eval.toMatrixString());
                    file.WriteLine();

                    // Get the confusion matrix
                    double[][] cmMatrixTest = eval.confusionMatrix();

                    System.Console.WriteLine("Bayesian Network results saved in Communication_Report.txt file successfully.");

                }
            }
        }

    }
}
