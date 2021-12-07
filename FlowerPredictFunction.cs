using System;
using System.IO;
using System.Net;
using System.Threading.Tasks;
using FlowerPredictionFunction.DataModels;
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
namespace FlowerPredictFunction
{
    public class FlowerPredictFunction
    {
        private readonly PredictionEnginePool<ImagePredictionInput, ImagePredictionOutput> _predictionEnginePool;

        public FlowerPredictFunction(PredictionEnginePool<ImagePredictionInput, ImagePredictionOutput> predictionEnginePool) => _predictionEnginePool = predictionEnginePool;

        [FunctionName("FlowerPredictFunction")]
        [OpenApiOperation(operationId: "Run", tags: new[] { "name" })]
        [OpenApiSecurity("function_key", SecuritySchemeType.ApiKey, Name = "code", In = OpenApiSecurityLocationType.Query)]
        [OpenApiParameter(name: "name", In = ParameterLocation.Query, Required = true, Type = typeof(string), Description = "The **Name** parameter")]
        [OpenApiResponseWithBody(statusCode: HttpStatusCode.OK, contentType: "text/plain", bodyType: typeof(string), Description = "The OK response")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Anonymous, "get", "post", Route = null)] HttpRequest req,
            ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            ImagePredictionInput image = JsonConvert.DeserializeObject<ImagePredictionInput>(requestBody);

            string modelFilePath = @"flowerClassifier.zip";
            var _environment = Environment.GetEnvironmentVariable("AZURE_FUNCTIONS_ENVIRONMENT");

            if (_environment == "Development")
            {
                modelFilePath = Path.Combine("flowerClassifier.zip");
            }
            else
            {
                modelFilePath = @"home/site/wwwroot/flowerClassifier.zip";
            }
            
            var mlContext = new MLContext(seed: 1);            
            var loadedModel = mlContext.Model.Load(modelFilePath, out var modelInputSchema);
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ImagePredictionInput, ImagePredictionOutput>(loadedModel);

            var prediction = predictionEngine.Predict(image);
            
            return new OkObjectResult(prediction);
        }
    }
}

