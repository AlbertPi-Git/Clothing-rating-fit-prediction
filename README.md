# Rating and fit prediction on RentTheRunWay dataset

This is the project of Web Mining and Recommender Systems course at UCSD. Rating and fit prediction with different models on RentTheRunWay dataset are conducted. Models are implemented with sklearn library and PyTorch.

## Acknowledgement
The dataset is provided by Prof. Julian McAuley at https://cseweb.ucsd.edu/~jmcauley/datasets.html.

## Single sample of dataset
Each sample in this dataset is a dict like the one below.<br>
{<br>
  &ensp; "fit": "fit",<br>
  &ensp; "user_id": "420272",<br>
  &ensp; "bust size": "34d",<br>
  &ensp; "item_id": "2260466",<br>
  &ensp; "weight": "137lbs",<br>
  &ensp; "rating": "10",<br>
  &ensp; "rented for": "vacation",<br>
  &ensp; "review_text": "An adorable romper! Belt and zipper were a little hard to navigate in a full day of wear/bathroom use, but <br>
  &ensp;&ensp;&ensp;&ensp;&ensp;&ensp; that's to be expected. Wish it had pockets, but other than that-- absolutely perfect! I got a million compliments.",<br>
  &ensp; "body type": "hourglass",<br>
  &ensp; "review_summary": "So many compliments!",<br>
  &ensp; "category": "romper",<br>
  &ensp; "height": "5' 8\"",<br>
  &ensp; "size": 14,<br>
  &ensp; "age": "28",<br>
  &ensp; "review_date": "April 20, 2016"<br>
}<br>

## Rating Prediction
Naive averaging, linear regression, ensemble of latent factor model and linear regression and shadow dense feed-forward neuron network are implemented and tested on the dataset. The performance of differet model are shown in the table below.

![rating performance](./Figs/rating_performance.png "Performance of different models on rating prediction")

## Fit Prediction
Logistic regression, SVM, random forest and gradient boosting are implemented and tested on the dataset. The performance of differet model are shown in the table below.

![fit performance](./Figs/fit_performance.png "Performance of different models on fit prediction")

## Detail
The detail of RentTheRunWay dataset and implementation can be seen in our PDF report.
