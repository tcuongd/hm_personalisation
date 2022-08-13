## H&M Personalized Fashion Recommendations

I began participating in this Kaggle competition: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations but didn't have time to properly tweak the modelling approach and produce a reasonable submission. It was still a good learning experience for me though:

1. I used [polars](https://www.pola.rs/) for the first time, a tabular data processing framework using the `arrow` backend with `pyspark`-like syntax and chaining. I really liked its lazy evaluation, where you can build a very complicated query plan, let `arrow` optimise it (e.g. via predicate pushdown), and execute once (preventing the creation of unncessary intermediate python objects like in `pandas`). This sped up exploratory analysis on the interactions dataset and allowed complicated transformations like windowing without running out of memory.
2. I built a matrix factorisation model using PyTorch (of course, only getting there after reading many articles) and now have a much better understanding of the concept of collaborative filtering. It was also my first time using PyTorch; the framework is very expressive, comprehensive, and easy to use in my opinion.

I don't think this modelling strategy would have ever performed well -- although there I implemented some interesting ideas like down-weighting interactions with discounted  items or that occurred a long time ago, from reading the [winning solution](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324070) it seems like I missed a few simple but key points:

* Having separate retrieval and ranking modules.
* For retrieval, focusing on simple "business rules" that maximise recall. These may just require aggregations of historical interaction data to build lookup tables per user.
* For ranking, using supervised ML that take simple user features and item features as inputs and produce a ranked list amongst the retrieved items.

### Installation

```sh
python -m pip install pipenv
pipenv install
```

#### Example - training a baseline matrix factorization model

```sh
pipenv run train_baseline
```

For other scripts, see `Pipfile`.
