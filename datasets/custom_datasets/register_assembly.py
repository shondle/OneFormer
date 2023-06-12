from detectron2.data import DatasetCatalog, MetadataCatalog
import os
from PIL import Image
import numpy as np
from torchvision import transforms

def get_files(query):
    # expects list for querying

    # appending all directory paths in the test_query
    label_dirs = []
    image_dirs = []
    for participant in query[0].split(", "):
        for dist in query[1].split(", "):
            for task in query[2].split(", "):
                for view in query[3].split(", "):
                    label_path = os.path.join('C:\\Users\\itsjo\\Documents\\repos\\assembly_glovebox_dataset', 'data', 'Labels', participant, dist, task, view)
                    image_path = os.path.join('C:\\Users\\itsjo\\Documents\\repos\\assembly_glovebox_dataset', 'data', 'images', participant, dist, task, view)
                    label_dirs.append(label_path)
                    image_dirs.append(image_path)
    return image_dirs, label_dirs


def assembly_dataset(image_dirs, label_dirs):

    # something here
    dataset = 10

    data_list = []

    for i, (image, label) in enumerate(zip(image_dirs, label_dirs)):
        image, label = image, label
        pil_image = Image.open(image)
        width, height = pil_image.size

        # need to call the PIL converter when loading the label - it is supposed to be grayscale -> done in the mapper

        data_dict = {
            "file_name": image,
            "height": 4,
            "width": 2,
            "image_id": i, # or is there something more distrinct I can use here
            "sem_seg_file_name": label,
        }
        data_list.append(data_dict)

    return data_list
if __name__ == '__main__':
    query = ['Test_Subject_1', 'ood', 'J', 'Top_View']
    image_dirs, label_dirs = get_files(query)


    DatasetCatalog.register("SemanticAssemblyDataset", assembly_dataset(image_dirs, label_dirs))
    MetadataCatalog.get("SemanticAssemblyDatset").stuff_classes = ["background", "left_hand", "right_hand"]
    MetadataCatalog.get("SemanticAssemblyDatset").evaluator_type = ["sem_seg"]

    # Later, to access the data,
    # data: List[Dict] = DatasetCatalog.get("SemanticAssemblyDataset")
    # ^ this is in the mapper method

