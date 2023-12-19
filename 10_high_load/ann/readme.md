# Seminar

```bash
# Clone repo
git clone https://github.com/clinc/oos-eval.git

# Create conda environment
conda env create -f environment.yml -n ann

# Compute real and synthetic embeddings
python prepare_embeddings.py

# Run benchmarks
python benchmarks.py
```

Подробнее почитать про бенчмарки других ANN можно [тут](https://github.com/erikbern/ann-benchmarks).
Про Faiss [тут](https://github.com/facebookresearch/faiss/wiki/).
