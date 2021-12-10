using System.Buffers.Text;
using System;
using System.IO;
using System.Net;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.Azure.WebJobs.Extensions.OpenApi.Core.Attributes;
using Microsoft.Azure.WebJobs.Extensions.OpenApi.Core.Enums;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.ML;
using Microsoft.ML;
using Microsoft.OpenApi.Models;
using Newtonsoft.Json;
using ModelTraining.DataModels;

namespace FlowerPredictFunction
{
    public class FlowerPredictFunction
    {
        private readonly PredictionEnginePool<ImagePredictionInput, ImagePredictionOutput> _predictionEnginePool;

        public FlowerPredictFunction(PredictionEnginePool<ImagePredictionInput, ImagePredictionOutput> predictionEnginePool) => _predictionEnginePool = predictionEnginePool;

        [FunctionName("FlowerPredictFunction")]
        [OpenApiOperation(operationId: "Run", tags: new[] { "name" })]
        [OpenApiRequestBody(contentType: "json", bodyType: typeof(string), Description = "Image",Example =typeof(Base64))]        
        [OpenApiResponseWithBody(statusCode: HttpStatusCode.OK, contentType: "text/plain", bodyType: typeof(ImagePredictionOutput), Description = "The OK response containing the prediction")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Anonymous, "post", Route = null)] HttpRequest req,
            ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            byte[] arr = Convert.FromBase64String(JsonConvert.DeserializeObject<string>(requestBody));
            ImagePredictionInput image = new ImagePredictionInput(arr, "label", "ImageFileName");

            string modelFilePath = @"flowerClassifier.zip";
            var _environment = Environment.GetEnvironmentVariable("AZURE_FUNCTIONS_ENVIRONMENT");

            if (_environment == "Development")
            {
                modelFilePath = Path.Combine("flowerClassifier.zip");
            }
            else
            {
                modelFilePath = $"{Environment.GetEnvironmentVariable("HOME")}/site/wwwroot/flowerClassifier.zip";
            }
            
            var mlContext = new MLContext(seed: 1);            
            var loadedModel = mlContext.Model.Load(modelFilePath, out var modelInputSchema);
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ImagePredictionInput, ImagePredictionOutput>(loadedModel);

            var prediction = predictionEngine.Predict(image);
            
            return new OkObjectResult(prediction);
        }
    }
}

