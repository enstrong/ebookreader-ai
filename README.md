# ebookreader-ai

This repository is the recommendation-system lab for `eBookReader`.

The main `ebookreader` project is the actual app: Flutter frontend, Spring Boot backend, PostgreSQL, books, reader, ratings, reviews, notes, audiobooks, all that civilized book-app stuff. This repository separates the AI work into its own place, because recommendation experiments get messy very quickly. Models, notebooks, checkpoints, sparse matrices, evaluation files, and "wait why is this MRR lower after two hours of training" thoughts deserve their own room.

The goal is simple:

```text
Given a user's reading history, recommend books they are likely to care about.
```

The actual path was less simple. It went from item-based collaborative filtering, to ALS, to hybrid ALS with metadata, and then to Neural Collaborative Filtering. This README focuses on the AI part only, especially:

- the ALS work from May
- the NCF work from July
- the fair comparison between ALS and NCF when metadata is removed from both sides

This is written as a project report, but in my style, because otherwise I am not publishing it and neither should you.

## Contents

1. [Project context](#project-context)
2. [Dataset](#dataset)
3. [Evaluation protocol](#evaluation-protocol)
4. [Metrics](#metrics)
5. [Part I: ALS](#part-i-als)
6. [Part II: NCF](#part-ii-ncf)
7. [NCF experiment sweep](#ncf-experiment-sweep)
8. [Best NCF model](#best-ncf-model)
9. [Fair ALS vs NCF comparison](#fair-als-vs-ncf-comparison)
10. [Conclusion](#conclusion)
11. [Future direction](#future-direction)
12. [References](#references)

## Project context

`eBookReader` started as my diploma project. The first big AI milestone was in May 2026: build a real recommendation engine instead of a toy "popular books" list wearing a fake mustache.

The first model family that became properly useful was ALS, short for Alternating Least Squares. That work happened in May, before the final diploma-defense sprint.

The NCF work happened later, in July 2026. There was a gap because, in order:

- I had to defend the diploma.
- I got the diploma with honors.
- I got into the AI Solutions program at FH Upper Austria.
- Then I finally had enough mental space to return to the model and ask: "Can a neural recommender beat my ALS champion?"

Very normal summer behavior. Some people go to the beach. I trained recommender models.

## Dataset

The recommendation experiments use the UCSD Goodreads dataset.

The important property of this dataset is that it is implicit-feedback shaped even when ratings exist. A row can mean that a user rated a book, read a book, reviewed a book, or interacted with it in another useful way.

The original app-side ALS work used a much larger prepared Goodreads interaction pipeline. The NCF repository works from a compact sparse matrix artifact:

```text
user_items.npz
```

That matrix is the common object used for the NCF experiments and the later fair ALS comparison.

The full Goodreads interaction source is much larger than a casual CSV. In the original ALS pipeline, the raw scan included:

| Dataset stage | Rows / interactions | Users | Books | Notes |
|---|---:|---:|---:|---|
| Raw rows scanned | 228,648,342 | - | - | Complete Goodreads interactions CSV |
| Explicit rating rows | 104,551,549 | - | - | Rows where `rating > 0` |
| Read-unrated rows | 7,579,654 | - | - | Rows where `rating == 0` and `is_read == 1` |
| Explicit-only filtered file | 99,361,816 | 750,325 | 724,641 | Users with at least 5 ratings, books with at least 10 ratings |
| Read-aware filtered file | 106,929,763 | 766,036 | 777,324 | Explicit ratings plus read-unrated rows |

Later, duplicate book editions were cleaned in the main project. That reduced the catalog from `777,324` books to `534,695` canonical books, which improved the final hybrid ALS model. This repository, however, is mainly about isolating the model behavior itself.

## Evaluation protocol

The core evaluation is leave-one-out recommendation.

For each evaluated user:

```text
1. Take one book from the user's history.
2. Hide it.
3. Train or score using the remaining user history.
4. Ask the model to rank candidate books.
5. Check where the hidden book appears.
```

This is the part that makes recommender evaluation feel almost rude. The model is handed a user's reading taste, one correct answer is quietly removed, and then the model has to find it again from a pile of books.

There are two evaluation modes:

| Evaluation mode | What happens | Why it matters |
|---|---|---|
| Sampled candidates | The hidden positive is ranked against sampled negatives, usually `500` total candidates. | Useful during training because it is fast. Scores are higher because the model has fewer books to compete against. |
| Full catalog | The hidden positive is ranked against the full `20,000`-book catalog used in this repository's matrix. | Much harder and much more honest. Scores are lower, as they should be. |

The fair ALS-vs-NCF comparison at the end uses the same matrix, same user-subset idea, same leave-one-out structure, and no metadata blending.

## Metrics

The main metrics are `Hit@K` and `MRR`.

**Hit@K** asks whether the hidden book appeared in the top `K` recommendations.

```text
Hit@10 = 0.104
```

means that the hidden book appeared somewhere in the top 10 recommendations for `10.4%` of evaluated users.

**MRR**, or Mean Reciprocal Rank, asks how early the hidden book appeared.

```text
rank 1  -> reciprocal rank 1.0
rank 2  -> reciprocal rank 0.5
rank 10 -> reciprocal rank 0.1
missing -> reciprocal rank 0
```

This is why MRR is usually the metric I care about most. A recommendation system is not just supposed to find the right book eventually. It should put the good stuff near the top. Nobody opens a book app thinking, "Yes, please show me the correct recommendation at position 67."

## Part I: ALS

ALS was the first model in this project that felt like real machine learning.

The model learns two matrices:

- a user-factor matrix
- a book-factor matrix

Each user and each book gets a vector of latent features. These are not manually named features like `fantasy`, `classics`, or `romance`. They are hidden taste dimensions learned from interaction patterns.

The score is simple:

```text
predicted preference = user_vector dot book_vector
```

So if a user vector and a book vector point in a similar direction, the model gives that book a higher score.

The ALS update for one user can be written as:

```text
A = Y.T @ Y + Y_obs.T @ (C - I) @ Y_obs + lambda * I
b = Y_obs.T @ (C * p)
x = solve(A, b)
```

Where:

- `Y` is the item-factor matrix
- `Y_obs` is the subset of item factors for books this user interacted with
- `C` is confidence
- `p` is preference
- `lambda` is regularization
- `x` is the solved user vector

ALS alternates between solving user vectors while item vectors are fixed, then solving item vectors while user vectors are fixed. It does this again and again until the vectors become useful.

The first ALS version used only 5-star ratings. That was clean, but too sparse. The model ignored too much signal.

The major improvement was mean-centering:

```text
centered_rating = rating - user's_average_rating
```

This matters because a rating does not mean the same thing for every person. A 3-star rating from someone who normally gives 5 stars is not the same as a 3-star rating from someone who usually gives 1 or 2 stars. People are inconsistent little rating machines, and the model has to respect that.

Then came the second improvement: read-aware implicit feedback.

In the Goodreads data, `rating == 0` does not mean "zero stars." It usually means the user read the book but did not leave a rating. Removing those rows throws away useful signal.

So the ALS pipeline treated:

```text
rating > 0:
  use mean-centered explicit rating signal

rating == 0 and is_read == 1:
  use weak positive implicit feedback
```

This follows the key idea from Hu, Koren, and Volinsky: preference and confidence are different things. Reading a book is evidence, but not the same evidence as loving it.

The strongest ALS-only model from the May work was:

```text
Read-aware mean-centered implicit ALS
features: 256
lambda: 1.0
iterations: 10
candidate books: 20,000
validation: true pre-training holdout
```

The strongest hybrid model in the main `ebookreader` repo added metadata reranking:

```text
final_score = alpha * ALS_score + (1 - alpha) * metadata_score
```

The best hybrid version used:

```text
Base model: ALS read-aware mean-centered
features: 256
lambda: 1.0
iterations: 10
alpha: 0.6
genre weight: 0.3
author weight: 0.5
rating weight: 0.1
page count weight: 0.05
popularity weight: 0.05
language weight: 0
candidate strategy: rerank ALS top 500
```

That hybrid ALS model became the champion in the main project:

| Model | Eval setup | Hit@5 | Hit@10 | Hit@20 | Hit@50 | MRR |
|---|---|---:|---:|---:|---:|---:|
| Item-CF, rating = 5 | old 5k restricted | 19.20% | 25.58% | 34.03% | 48.38% | 0.1494 |
| ALS read-aware 256f lambda 1.0 10i | true validation 20k | 23.73% | 29.94% | 36.85% | 47.66% | 0.1779 |
| Hybrid ALS + metadata | true validation 20k | 24.99% | 30.94% | 37.89% | 48.90% | 0.1869 |
| Hybrid ALS + metadata on cleaned dataset | true validation 20k | 27.37% | 33.88% | 41.48% | 51.78% | 0.2143 |

The important lesson from ALS was not "matrix factorization magically wins." The important lesson was that data representation wins first.

Mean-centering helped. Keeping read-unrated rows helped. Cleaning duplicate editions helped. Only after those decisions did the model become genuinely strong.

## Part II: NCF

After the diploma work, I moved the neural recommender experiments into this repository.

NCF stands for Neural Collaborative Filtering. The basic idea is still to learn user and item embeddings, but instead of scoring them only with a dot product, a neural network learns the interaction function.

The model in this repository uses:

- a learned user embedding
- a learned book embedding
- concatenation of both embeddings
- a multilayer perceptron
- dropout
- sigmoid output
- binary cross-entropy loss
- negative sampling

In simplified form:

```text
user_vec = user_embedding(user_id)
book_vec = book_embedding(book_id)
x = concatenate(user_vec, book_vec)
score = MLP(x)
```

ALS says:

```text
user dot book
```

NCF says:

```text
let a neural network learn how this user vector and this book vector should interact
```

That is more flexible. It is also more expensive, more fragile, and much more willing to punish me for not having enough compute.

The training data is built from positives and sampled negatives:

```text
positive pair: user interacted with this book -> label 1
negative pair: sampled unread book -> label 0
```

The main negative ratio was:

```text
4 negatives per positive
```

So each positive interaction creates one positive training row and four negative training rows.

## NCF experiment sweep

The July experiment sweep tested different user counts, embedding sizes, layer sizes, learning rates, and training durations.

The main grid included:

- users: `5,000`, `20,000`, `50,000`
- embedding dimensions: `64`, `128`, `256`, `512`
- layers:
  - `[128, 64, 32]`
  - `[256, 128, 64]`
  - `[512, 256, 128]`
  - `[1024, 512, 256]`
- dropout: `0.2`
- learning rates: `0.0003`, `0.001`, `0.003`, `0.01`
- batch sizes: `1024`, `4096`
- negative ratio: `4`
- epochs: mostly `25`

In total, the logged spreadsheet contains 21 experiment rows, including the early baselines, 5k hyperparameter runs, larger-user runs, and one interrupted 50k full evaluation.

The first weird-but-important lesson was that sampled evaluation can look very good while full-catalog evaluation stays much more modest. This is expected. Ranking one hidden book among 500 candidates is not the same sport as ranking it among 20,000 books.

The second lesson was that lower training loss did not automatically mean better full-catalog ranking. Some of the larger or higher-learning-rate models looked strong in sampled evaluation but collapsed when asked to rank against the full catalog.

That is the recommender-system version of "great in rehearsals, questionable on stage."

## Best NCF model

The best NCF model from the July sweep was:

```text
Model: ncf_5k_256_lr3e3
Date: 17.07.2026
Users: 5,000
Embedding: 256
Layers: [512, 256, 128]
Dropout: 0.2
Learning rate: 0.003
Batch size: 1024
Negative ratio: 4
Epochs: 25
Best epoch: 10
Training time: 43m 1s
Final loss: 0.162
Sampled MRR@500: 0.5178
Sampled Hit@10@500: 0.582
Full Hit@10: 0.104
Full MRR: 0.0658
```

As a table:

| Date | Model | Users | Embedding | Layers | Dropout | LR | Batch | NegRatio | Epochs | Training Time | Final Loss | Hit@10 (500) | MRR (500) | Full Hit@10 | Full MRR |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17.07.2026 | ncf_5k_256_lr3e3 | 5,000 | 256 | [512,256,128] | 0.2 | 0.003 | 1024 | 4 | 25 | 43m 1s | 0.162 | 0.582 | 0.5178 | 0.104 | 0.0658 |

The best full-catalog NCF result came from the 256-dimensional model trained on 5,000 users. It did not have the lowest loss in the whole sweep. It was not the largest model. It was not the one with the most heroic training time.

It was simply the best at putting the held-out book near the top when evaluated against the full 20k catalog.

That matters more than looking impressive in the model name.

## Fair ALS vs NCF comparison

The original ALS champion in the main app is a hybrid system. It mixes collaborative filtering with metadata like author, genre, rating, page count, popularity, and language.

That is useful in the product, but it is not a fair model-only comparison with NCF.

So for this repository, I compared ALS and NCF under a stricter rule:

```text
No hybrid metadata.
No author boost.
No genre boost.
No product-side reranking.
Just the model behavior.
```

The ALS comparison model uses the same `user_items.npz` matrix and an NCF-style leave-one-out protocol. It trains plain implicit ALS on the same user subset sizes and evaluates with both sampled candidates and the full catalog.

Here are the fair comparison results:

| Model | Users | Sampled Hit@10 | Sampled MRR | Full Hit@5 | Full Hit@10 | Full Hit@20 | Full Hit@50 | Full MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NCF 5k champion | 5,000 | 0.859 | 0.5178 | 0.076 | 0.104 | 0.162 | 0.248 | 0.0658 |
| ALS fair 5k | 5,000 | 0.469 | 0.2103 | 0.034 | 0.048 | 0.066 | 0.160 | 0.0234 |
| NCF 50k | 50,000 | 0.649 | 0.3571 | 0.084 | 0.114 | 0.178 | 0.304 | 0.0657 |
| ALS fair 50k | 50,000 | 0.604 | 0.2967 | 0.034 | 0.052 | 0.102 | 0.198 | 0.0286 |

Under this model-only protocol, NCF beats plain ALS clearly.

So the neural model is better than plain ALS when both are stripped down to collaborative behavior only. But the full project champion is still the cleaned hybrid ALS model with metadata, at `MRR = 0.2143`. This is not a contradiction. It basically means:

```text
NCF > plain ALS
ALS + metadata > current NCF
```

The obvious next experiment is:

```text
NCF + full dataset + metadata
```

And this is where compute enters the room, pulls up a chair, and quietly ruins the evening.

The current NCF work was constrained by available hardware. The 21 different experiments took 24+ hours to train on 5k-50k users. Training the neural model on the full 685k-ish user-scale dataset, while also mixing in metadata features, would require much more time and compute than I had available during this experiment cycle.

The theoretical expectation is that a full-scale metadata-aware NCF model should outperform the ALS champion. It has the model flexibility, it already beats plain ALS in the fair setup, and metadata was exactly what made ALS much stronger in the production-style pipeline.

But "theoretically should" is not a result. It's just a hypothesis.

For now, the honest conclusion is:

- current best product model: cleaned hybrid ALS
- current best pure collaborative model in this repo: NCF
- most promising future model: metadata-aware NCF trained on the full dataset

## Conclusion

The recommendation work reached three useful conclusions.

First, ALS is extremely strong when the data is represented well. Mean-centering, implicit read signals, duplicate cleanup, and metadata reranking mattered more than just increasing model size.

Second, NCF has much more room to grow than ALS. ALS is powerful, but its scoring function is still fundamentally a dot product between a user vector and a book vector. NCF can learn more complicated user-book correlations through the neural network on top of the embeddings. That is exactly why it beat plain ALS in the fair comparison, and also exactly why it is expensive. The same flexibility that gives NCF more potential also makes it hungry for data, epochs, tuning, and compute. I do think a full-scale metadata-aware NCF model should eventually beat the ALS champion, but I cannot honestly claim that result yet because I do not currently have the computational power to train it properly.

Third, future me should be less heroic and more strategic with experiments. Start with iterations or epochs first, because knowing when the model stops improving saves a ridiculous amount of time. Use a real train/validation split from the beginning, otherwise the metrics become too optimistic and then you have to emotionally recover from your own spreadsheet. Test parameters like `lambda` and learning rate on a log scale, because `0.1 -> 0.2` can do almost nothing while `0.01 -> 0.1 -> 1.0 -> 10.0` actually tells you something. Run hyperparameter experiments on smaller data first, then spend the expensive full-dataset training runs only on the combinations that already proved themselves.

The current winner depends on the question:

| Question | Winner |
|---|---|
| Best model-only collaborative recommender in this repo | NCF |
| Best practical recommender built so far for eBookReader | Hybrid ALS + metadata |
| Most promising next research direction | Full-scale metadata-aware NCF |

That is a satisfying result. Not finished, but satisfying. The kind of result where the model did not solve everything, but it did make the next door very obvious.

## Future direction

For recommendation systems, the next big step would be training NCF on the full dataset and adding metadata features directly into the model.

A future version could use:

- user embeddings
- book embeddings
- author embeddings
- genre/category features
- rating/popularity features
- page count and language features
- maybe text embeddings from descriptions, if I decide to make my laptop suffer artistically

But for now, I am finally moving into computer vision.

I am not going to say what the project is about yet. But cardio will definitely become more fun to me. The project should be fun and useful at the same time, which is what makes me excited about it.

## References

1. Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets*. Proceedings of the 2008 IEEE International Conference on Data Mining.

2. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). *Neural Collaborative Filtering*. Proceedings of the 26th International Conference on World Wide Web (WWW 2017).

3. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback*. Proceedings of UAI 2009.

4. Wan, M., & McAuley, J. (2018). *Item Recommendation on Monotonic Behavior Chains*. Proceedings of the 12th ACM Conference on Recommender Systems (RecSys 2018).

5. Wan, M., Misra, R., Nakashole, N., & McAuley, J. (2019). *Fine-Grained Spoiler Detection from Large-Scale Review Corpora*. Proceedings of ACL 2019.

6. Wan, M., & McAuley, J. UCSD Goodreads Dataset. https://github.com/MengtingWan/goodreads
