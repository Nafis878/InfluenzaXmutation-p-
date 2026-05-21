# Supplementary: Agentic Drift Alignment Monitoring

This directory contains `agentic_drift.py`, which is **not part of the main
InfluenzaXmutation submission** and is excluded from the primary pipeline.

## AI Alignment Extension — Out of Scope for Current Submission

The `agentic_drift.py` module applies the biological divergence mathematics
developed for influenza HA evolution to the problem of detecting behavioral
drift in LLM-based autonomous agents. While conceptually related to the
biological drift framework in the main pipeline, this analysis:

1. Operates on simulated agent lifecycle data, not experimentally validated
   biological sequences.
2. Uses proxy behavioral metrics that lack the experimental grounding of
   WHO/CDC HI assay validation.
3. Addresses an AI safety research question that is distinct from the
   virological scope of this submission.

This module is **reserved for a separate future publication** focused on
AI alignment monitoring. Reviewers of the main InfluenzaXmutation paper
should treat this directory as supplementary background context only.

## Running (standalone)

```bash
cd supplementary
python agentic_drift.py
```

Outputs go to `../agentic_drift_models/`.

## Citation

If this extension is used independently, please cite the companion manuscript
(in preparation) alongside the main InfluenzaXmutation paper.
