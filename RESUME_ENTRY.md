# clipsearch — Evaluating AI Image Search
Tech: Python, PyTorch, OpenCLIP, NumPy, Matplotlib, pytest, GitHub Actions
- Evaluated text-to-image search on 5,003 real COCO queries, comparing keyword tags (6.8% recall@1), zero-shot CLIP (49.7%), a keyword/CLIP hybrid and a trained adapter, with cluster-bootstrap 95% confidence intervals.
- Trained a contrastive adapter on frozen CLIP embeddings with early stopping, raising recall@1 by 2.4 points (95% CI 0.6–4.1) and by 6.9 points on counting queries, CLIP's weakest category.
- Broke results down by query type (counting, spatial, negation, text in image) with paired tests, and caught a silent model-config mismatch that skewed embeddings.
