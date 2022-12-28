# OralCancer-MachineLearning
## Machine Learning Project

- There are several potential benefits to using machine learning for oral cancer detection. One benefit is that machine learning algorithms can potentially identify patterns that are not easily recognizable by humans, increasing the accuracy of the diagnosis. 
- Another benefit is that machine learning algorithms can potentially process large amounts of data more quickly and efficiently than humans, making the detection process more efficient.

### Requirements : 
    requirments.txt
- opencv-python
- numpy
- Flask
- pandas
- scikit-learn

### Model :
    model.py  -> Python file
    model.pkl -> Pickle file
- Support Vector Classifier was used to train the model.
- Then the trained model is saved using pickle.

### Use of Flask :
    templates/index.html -> HTML
    app.py               -> Python file
- Flask was used as framework for creating a user interface where the user uploads the image and receives back the predicted output.
