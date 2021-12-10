using System.IO;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML;
using ModelTraining.DataModels;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;

namespace ModelTraining
{
    class Program
    {
        static void Main(string[] args)
        {
            const string assetsRelativePath = @"C:\Users\luisf\Downloads\Machine Learning personal dataset (2) - Cópia\Machine Learning personal dataset";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            string outputMlNetModelFilePath = Directory.GetParent(Environment.CurrentDirectory.ToString()) +  @"\FlowerPredictionFunction\MLModels\flowerClassifier.zip";
            System.Console.WriteLine(outputMlNetModelFilePath
            );
            string imagesFolderPathForPredictions = @"C:\Users\luisf\Downloads\TestImages";


            var mlContext = new MLContext(seed: 1);
            mlContext.Log += FilterMLContextLog;           

            // 1. Load the initial full image-set into an IDataView and shuffle so it'll be better balanced
            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);
            IDataView fullImagesDataset = mlContext.Data.LoadFromEnumerable(images);
            IDataView shuffledFullImageFilePathsDataset = mlContext.Data.ShuffleRows(fullImagesDataset);

            // 2. Load Images with in-memory type within the IDataView and Transform Labels to Keys (Categorical)
            IDataView shuffledFullImagesDataset = mlContext.Transforms.Conversion.
                    MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.LoadRawImageBytes(
                                                outputColumnName: "Image",
                                                imageFolder: assetsRelativePath,
                                                inputColumnName: "ImagePath"))
                .Fit(shuffledFullImageFilePathsDataset)
                .Transform(shuffledFullImageFilePathsDataset);

            // 3. Split the data 80:20 into train and test sets, train and evaluate.
            var trainTestData = mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.2);
            IDataView trainDataView = trainTestData.TrainSet;
            IDataView testDataView = trainTestData.TestSet;

            // 5. Define the model's training pipeline using DNN default values
            //
            var pipeline = mlContext.MulticlassClassification.Trainers
                    .ImageClassification(featureColumnName: "Image",
                                         labelColumnName: "LabelAsKey",
                                         validationSet: testDataView)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel",
                                                                      inputColumnName: "PredictedLabel"));

            // 5.1 (OPTIONAL) Define the model's training pipeline by using explicit hyper-parameters
            //
            //var options = new ImageClassificationTrainer.Options()
            //{
            //    FeatureColumnName = "Image",
            //    LabelColumnName = "LabelAsKey",
            //    // Just by changing/selecting InceptionV3/MobilenetV2/ResnetV250  
            //    // you can try a different DNN architecture (TensorFlow pre-trained model). 
            //    Arch = ImageClassificationTrainer.Architecture.MobilenetV2,
            //    Epoch = 50,       //100
            //    BatchSize = 10,
            //    LearningRate = 0.01f,
            //    MetricsCallback = (metrics) => Console.WriteLine(metrics),
            //    ValidationSet = testDataView
            //};

            //var pipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(options)
            //        .Append(mlContext.Transforms.Conversion.MapKeyToValue(
            //            outputColumnName: "PredictedLabel",
            //            inputColumnName: "PredictedLabel"));

            // 6. Train/create the ML model
            Console.WriteLine("*** Training the image classification model with DNN Transfer Learning on top of the selected pre-trained model/architecture ***");


            //Train
            ITransformer trainedModel = pipeline.Fit(trainDataView);

            // 7. Get the quality metrics (accuracy, etc.)
            EvaluateModel(mlContext, testDataView, trainedModel);

            // 8. Save the model to assets/outputs (You get ML.NET .zip model file and TensorFlow .pb model file)
            mlContext.Model.Save(trainedModel, trainDataView.Schema, outputMlNetModelFilePath);
            Console.WriteLine($"Model saved to: {outputMlNetModelFilePath}");

            // 9. Try a single prediction simulating an end-user app
            TrySinglePrediction(imagesFolderPathForPredictions, mlContext, trainedModel);
        }
       
        private static void EvaluateModel(MLContext mlContext, IDataView testDataset, ITransformer trainedModel)
        {
            Console.WriteLine("Making predictions in bulk for evaluating model's quality...");

            // Measuring time
            var watch = Stopwatch.StartNew();

            var predictionsDataView = trainedModel.Transform(testDataset);

            var metrics = mlContext.MulticlassClassification.Evaluate(predictionsDataView, labelColumnName:"LabelAsKey", predictedLabelColumnName: "PredictedLabel");
            watch.Stop();
            var elapsed2Ms = watch.ElapsedMilliseconds;

            Console.WriteLine($"Predicting and Evaluation took: {elapsed2Ms / 1000} seconds");            
            Console.WriteLine($"Micro accuracy: {metrics.MicroAccuracy}");
            Console.WriteLine($"Macro accuracy: {metrics.MacroAccuracy}");
            Console.WriteLine($"ConfusionMatrix: {metrics.ConfusionMatrix.ToString()}");            
        }

        private static void TrySinglePrediction(string imagesFolderPathForPredictions, MLContext mlContext, ITransformer trainedModel)
        {
            // Create prediction function to try one prediction
            var predictionEngine = mlContext.Model
                .CreatePredictionEngine<ImagePredictionInput, ImagePredictionOutput>(trainedModel);
            var testImages = FileUtils.LoadLocalImages(
                imagesFolderPathForPredictions, false);
            var imageToPredict = testImages.First();
            var prediction = predictionEngine.Predict(imageToPredict);

            Console.WriteLine(
                $"Image Filename : [{imageToPredict.ImageFileName}], " +
                $"Scores : [{string.Join(",", prediction.Score)}], " +
                $"Predicted Label : {prediction.PredictedLabel}");
        }


        public static IEnumerable<ImageData> LoadImagesFromDirectory(
            string folder,
            bool useFolderNameAsLabel = true)
            => FileUtils.LoadImagesFromDirectory(folder, useFolderNameAsLabel)
                .Select(x => new ImageData(x.imagePath, x.label));

        public static string GetAbsolutePath(string relativePath)
            => FileUtils.GetAbsolutePath(typeof(Program).Assembly, relativePath);


        private static void FilterMLContextLog(object sender, LoggingEventArgs e)
        {
            if (e.Message.StartsWith("[Source=ImageClassificationTrainer;"))
            {
                Console.WriteLine(e.Message);
            }
        }
    }
}
