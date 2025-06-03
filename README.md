# MOFormer: Navigating the Antimicrobial Peptide Design Space with Pareto-based Multi-Objective Transformer
learning holds the potential to revolutionize antibiotic development. Despite recent progress in AMP generation, designing peptide antibiotics with multiple optimal properties remains a significant challenge. We present MOFormer, an advanced multi-objective AMP design pipeline capable of optimizing multiple AMP properties simultaneously. By leveraging a conditional Transformer, the model refines the AMP sequence-property landscape for efficient multi-objective generation. It also incorporates regularization techniques to maintain a highly structured space, enabling the sampling of precise and desirable candidates. Comparative analyses reveal that MOFormer achieves the optimal hypervolume in the multi-objective space, surpassing advanced methods in simultaneously maximizing antimicrobial activity (minimum inhibitory concentration, MIC) and minimizing hemolysis (HEMO) and toxicity (TOXI), thereby yielding the most promising and desirable set of candidate peptides. When extended to a tri-objective scenario, MOFormer continues to exhibit remarkable optimization performance. Finally, we execute a hierarchical and rapid ranking of generated candidates based on Pareto Fronts. We conducted a comprehensive validation of the physicochemical properties and target attributes of the candidates, while AlphaFold structure predictions revealed notably reliable predicted local distance difference test(PIDDT) scores ranging from 70\% to 87\%. Our findings suggest that MOFormer holds potential to accelerate the discovery of efficacious peptide antibiotics by optimizing multi-objective trade-offs.


# Requirements
```PYTHON
conda env create -f MOFormer.yaml
```

# Steps
1. Prepare datasets（data folder,PCA.py）
2. Train MOFormer model (train.py)
3. Generate candidates (inference.py)
4. Compute pareto front(HV.py)
5. Analysis results (analysis folder)





