Pothole Detection with Detectron2

Dataset Link: https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset

About Dataset:
A Fully Annotated Image Dataset for Pothole Detection

Fig: Distribution of Different Categories of Potholes

Pothole Size-Categories
Small: BoundingBox Area <= 1024px
Medium: 1024px < BoundingBox Area <= 9216px
Large: BoundingBox Area > 9216px
Note: These size-categories were calculated after resizing the images to 300x300 pixels keeping the aspect ratio. This is similar to Microsoft COCO Size Metrics.

Splits
The directory annotated-images contains the images having pothole and their respective annotations (as XML file).
The file splits.json contains the annotation filenames (.xml) of the training (80%) and test (20%) dataset in following format---

{
  "train": ["img-110.xml", "img-578.xml", "img-455.xml", ...],
  "test": ["img-565.xml", "img-498.xml", "img-143.xml", ...]
}



