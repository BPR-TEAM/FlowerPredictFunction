using Microsoft.ML.Data;

namespace FlowerPredictionFunction.DataModels{
     public class ImagePredictionOutput
    {
        [ColumnName("Score")]
        public float[] Score;

        [ColumnName("PredictedLabel")]
        public string PredictedLabel;
    }
}