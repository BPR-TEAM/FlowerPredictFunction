using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using ModelTraining.DataModels;

namespace ModelTraining
{
    public class FileUtils
    {
        public static IEnumerable<(string imagePath, string label)> LoadImagesFromDirectory(
            string folder,
            bool useFolderNameasLabel)
        {
            var imagesPath = Directory
                .GetFiles(folder, "*", searchOption: SearchOption.AllDirectories)
                .Where(x => Path.GetExtension(x) == ".jpg" || Path.GetExtension(x) == ".png");

            return useFolderNameasLabel
                ? imagesPath.Select(imagePath => (imagePath, Directory.GetParent(imagePath).Name))
                : imagesPath.Select(imagePath =>
                {
                    var label = Path.GetFileName(imagePath);
                    for (var index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                    return (imagePath, label);
                });
        }

        public static IEnumerable<ImagePredictionInput> LoadLocalImages(
            string folder,
            bool useFolderNameAsLabel = true)
        {
            return LoadImagesFromDirectory(folder, useFolderNameAsLabel)
                .Select(x => new ImagePredictionInput(
                    image: File.ReadAllBytes(x.imagePath),
                    label: x.label,
                    imageFileName: Path.GetFileName(x.imagePath)));
        }

        public static string GetAbsolutePath(Assembly assembly, string relativePath)
        {
            var assemblyFolderPath = new FileInfo(assembly.Location).Directory.FullName;

            return Path.Combine(assemblyFolderPath, relativePath);
        }
    }
}