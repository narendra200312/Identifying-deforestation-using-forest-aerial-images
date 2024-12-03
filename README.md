# Identifying-deforestation-using-forest-aerial-images
I have implemented U-Net and sequential model for segmentation and classification of the satellite images. I have trained the model's weights and saved the with the names "forest_segmentation_model.h5" and "segmentation.h5". And to represent the results of my project with the help of streamlit app "app2.py" by recreating the models architecture and model weights that have been saved.

Dataset contains :
The dataset is consisting of 5108 satellite images and their respective binary mask images. And also a meta_data.csv file containing names of the images and their masks names with the column names "images" and "masks".

Acknowledgment :
The dataset is obtained from the "DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images".
I. Demir et al., "DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Salt Lake City, UT, USA, 2018, pp. 172-17209, doi: 10.1109/CVPRW.2018.00031. keywords: {Satellites;Roads;Building management systems;Climate change;Urban areas;Computer vision;Image segmentation;Smart cities},

