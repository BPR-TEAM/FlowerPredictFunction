namespace ModelTraining.DataModels{
   public class ImagePredictionInput
    {
        public ImagePredictionInput()
        {
        }

        public ImagePredictionInput(byte[] image, string label, string imageFileName)
        {
            Image = image;
            Label = label;
            ImageFileName = imageFileName;
        }

        public byte[] Image {get;set;}

        public string Label {get;set;}

        public string ImageFileName {get;set;}
    }
}