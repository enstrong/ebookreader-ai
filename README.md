# Book recommendation experiments

I built a recommendation engine for [eBookReader](https://github.com/enstrong/ebookreader), my Flutter and Spring Boot diploma project, then continued the model work here. The question was practical: given a person's reading history, can a model rank a book they would want to read near the top of a large catalog?

I tested item-based collaborative filtering, implicit ALS, a metadata-reranked ALS system, and Neural Collaborative Filtering (NCF). The best app-side system so far is the **cleaned ALS + metadata hybrid: 33.88% Hit@10 and 0.2143 MRR** in its 20,000-book validation setup. In a separate, matched model-only comparison, **NCF beat plain ALS** on the same sparse interaction matrix. These are different experiments; their scores should not be compared as if they came from one shared test.

The [full experiment notes](docs/experiment-notes.md) retain the data-processing history, hyperparameter sweep, and detailed results. This page is the short path through the project.

## Results at a glance

| Question | Evaluation | Result |
| --- | --- | ---: |
| Best recommender built for eBookReader | Cleaned hybrid ALS + metadata; held-out books, 20,000-book catalog | **33.88% Hit@10; 0.2143 MRR** |
| Does NCF beat plain ALS? | Same interaction matrix and model-only protocol; 5,000-user training subset, 20,000-book catalog | **10.4% vs 4.8% Hit@10** |
| Does that result hold at a larger subset? | Same model-only protocol; 50,000-user training subset | **11.4% vs 5.2% Hit@10** |

**Hit@10** is the share of evaluated users whose hidden book appeared among the top ten recommendations. **MRR** rewards finding that book higher in the ranking. These are offline ranking measures, not a claim that one third of real readers would like every suggested book.

The two comparison rows isolate model behavior by removing metadata reranking from both sides. The hybrid result uses additional item metadata and a cleaned catalog, so its 33.88% should not be read as a head-to-head score against NCF's 10.4%.

## What I built

```text
Goodreads interactions
  → filter users and books; preserve read-but-unrated events
  → hold out one positive book per evaluated user
  → train item-CF / implicit ALS / NCF
  → optionally rerank ALS candidates with book metadata
  → rank against sampled candidates during iteration or 20,000 books for final evaluation
```

- **Item-CF** was the baseline: recommend books similar to those a user interacted with.
- **Implicit ALS** learned user and book factors. I improved the signal by centering explicit ratings around each user's average and treating read-but-unrated books as weaker positive feedback.
- **Hybrid ALS** reranked ALS candidates with author, genre, rating, page count, and popularity features. Cleaning duplicate editions improved this app-side system further.
- **NCF** learned user and book embeddings plus a multilayer perceptron scoring function. I ran 21 logged experiments across training-subset sizes, embedding widths, learning rates, and layer sizes.

The raw Goodreads scan contained **228,648,342 interaction rows**. That is the input scanned, not the size of any single model's training set. After filtering, the read-aware interaction file contained **106,929,763 rows**; the NCF comparison used a smaller sparse matrix and 5,000- or 50,000-user training subsets.

## Evaluation

For each evaluated user, I held out one book from their history and asked the model to rank it using the remaining history. Fast training checks ranked it against sampled negatives. Final reported comparisons ranked it against the full **20,000-book evaluation catalog**, which is much harder.

### App-side ALS and hybrid validation

| Model | Hit@10 | MRR |
| --- | ---: | ---: |
| Read-aware ALS, 256 factors | 29.94% | 0.1779 |
| ALS + metadata reranking | 30.94% | 0.1869 |
| ALS + metadata, cleaned editions | **33.88%** | **0.2143** |

These figures come from the main app-side validation pipeline. The earlier item-CF baseline used a restricted 5,000-book evaluation and therefore is omitted from this table.

### Matched model-only comparison

| Training users | Model | Full-catalog Hit@10 | Full-catalog MRR |
| ---: | --- | ---: | ---: |
| 5,000 | NCF | **10.4%** | **0.0658** |
| 5,000 | Plain ALS | 4.8% | 0.0234 |
| 50,000 | NCF | **11.4%** | **0.0657** |
| 50,000 | Plain ALS | 5.2% | 0.0286 |

Both models in this table use the same sparse interaction matrix, leave-one-out structure, and no metadata boost. The 50,000-user NCF run is stored in a results file whose historical filename starts with `ncf_20k`; the `n_users_train` field inside that file is **50,000**.

## What the experiments taught me

The largest gain in the app-side pipeline came from **representing the data better**: user-relative ratings, read-without-rating events, and duplicate-edition cleanup. More model complexity was not automatically better.

NCF performed better than plain ALS in the matched comparison, but it did not replace the cleaned hybrid ALS system used by the app. The next model experiment would add metadata to NCF and evaluate it under the **same split and catalog** as the hybrid. Until then, claiming that one model is universally better would go beyond the results.

## Explore the work

| Path | What it contains |
| --- | --- |
| [`notebooks/ebookreader_ncf.ipynb`](notebooks/ebookreader_ncf.ipynb) | NCF exploration and training |
| [`notebooks/2026_07_19-20k-champmodel.ipynb`](notebooks/2026_07_19-20k-champmodel.ipynb) | Later NCF experiment |
| [`scripts/recommendations/`](scripts/recommendations/) | Interaction preparation, validation splits, ALS training, and hybrid evaluation |
| [`results/`](results/) | Logged NCF JSON results and comparison workbooks |
| [`docs/experiment-notes.md`](docs/experiment-notes.md) | Full methods, parameters, and experiment history |

To inspect a recorded NCF run without installing ML packages:

```bash
python -m json.tool results/ncf_5k_256_lr3e3_results.json
```

The repository includes code and result logs, but **does not include the full Goodreads data, derived sparse matrix, or trained model artifacts**. Those are large external inputs. It also does not yet have a pinned environment file or a one-command reproduction pipeline. The scripts and notebooks show the methods; an exact rerun requires obtaining the [UCSD Goodreads dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home), preparing the matrices, and recreating the recorded environment and split. The [experiment notes](docs/experiment-notes.md) explain the data stages and evaluation choices.

## Limits and next work

- The reported metrics measure ranking on held-out interactions. They do not measure reader satisfaction or real-world click-through.
- A 20,000-book evaluation catalog is smaller than the raw Goodreads catalog. Catalog size and candidate sampling affect Hit@K.
- The neural runs used at most 50,000 training users, well below the full filtered user set.
- The current repository needs pinned dependencies, data-preparation instructions, and a small public test fixture before another person can rerun it end to end.
- Cold-start readers and books, popularity bias, and behavior after deployment remain open questions.

My next useful step is a reproducible, shared evaluation pipeline for hybrid ALS and metadata-aware NCF. That would answer the comparison the current results cannot.

## References

- Hu, Koren, and Volinsky, [Collaborative Filtering for Implicit Feedback Datasets](https://ieeexplore.ieee.org/document/4781121), 2008.
- He et al., [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031), 2017.
- Wan and McAuley, [UCSD Book Graph / Goodreads dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home).
