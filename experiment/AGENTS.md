Write me end_to_end.ipynb that combines the idea from [detection2.ipynb](experiment/detection2.ipynb) and [new2.ipynb](experiment/new2.ipynb)

I want a two stage training separated by sections:

# Data split
First split the data early on i.e which is for training, val, and test

# Section 1 (Fault Detection) 
- The model is trained on all operating conditions in the provided dataset, the goal is that the model learns what is a "known" operating condition. Show the confusion matrix of the known vs unknown 
- Use the Mahalanobis-based method 


# Section 2 (Fault Classifier)
- Use the CNN from new2.ipynb, trained in the same method. Show the confusion matrix of the labelled output


# Evaluation
- Combine the two stages i.e first the fault detector short-circuits all unknown conditions then known conditions is passed to downstream classifier. In the final output report the confusion matrix with all classes + unknown

