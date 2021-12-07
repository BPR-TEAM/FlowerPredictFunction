using System;
using System.IO;
using FlowerPredictionFunction.DataModels;
using FlowerPredictionFunction;
using Microsoft.Azure.Functions.Extensions.DependencyInjection;
using Microsoft.Extensions.ML;

[assembly: FunctionsStartup(typeof(Startup))]
namespace  FlowerPredictionFunction
{
    public class Startup : FunctionsStartup
    {
        private readonly string _environment;
        private readonly string _modelPath;

        public Startup()
        {
            _environment = Environment.GetEnvironmentVariable("AZURE_FUNCTIONS_ENVIRONMENT");

            if (_environment == "Development")
            {
                _modelPath = Path.Combine("FlowerClassification.Train", "flowerClassifier.zip");
            }
            else
            {
                string deploymentPath = @"D:\home\site\wwwroot\";
                _modelPath = Path.Combine(deploymentPath,"flowerClassifier.zip");
            }
        }

        public override void Configure(IFunctionsHostBuilder builder)
        {
            builder.Services.AddPredictionEnginePool<ImagePredictionInput, ImagePredictionOutput>()
                .FromFile(modelName: "FlowerClassifier", filePath: _modelPath, watchForChanges: true);
        }
    }
}