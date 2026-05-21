# Data Dictionary
Generated: 2026-05-22T01:09:04.551850

## phase1_h1n1_filtered.csv
| Column | Type | Description |
|--------|------|-------------|
| Accession | str | GenBank accession number |
| Host | str | Host species (Human, Avian, etc.) |
| Subtype | str | Influenza subtype (H1N1) |
| Year | int | Collection year |
| Sequence | str | Amino acid or nucleotide sequence |
| Length | int | Sequence length in characters |
| Country | str | Country of collection |
| VirusName | str | Full virus strain name |
| Protein | str | Protein type (HA, etc.) |

## phase1_h1n1_baseline_statistics.csv
| Column | Type | Description |
|--------|------|-------------|
| Metric | str | Statistic name (e.g. Length_mean) |
| Value | float | Computed value |

## phase1_h1n1_divergence_rates.csv
| Column | Type | Description |
|--------|------|-------------|
| Year | int | Collection year |
| mean_distance | float | Mean Hamming distance to reference |
| std_distance | float | Standard deviation of distance |
| sample_size | int | Number of sequences in that year |

## phase2_h3n2_filtered.csv
Same schema as phase1_h1n1_filtered.csv but for H3N2 sequences at mode length.

## phase2_h3n2_historical_clusters.csv
Same as phase2_h3n2_filtered.csv plus:
| historical_cluster | str | Assigned cluster name (HK68, EN72, …) |

## phase2_h3n2_silhouette_scores.csv
| Column | Type | Description |
|--------|------|-------------|
| K | int | Number of clusters tested |
| silhouette_score | float | Average silhouette score for this K |

## phase2_h3n2_cluster_assignments.csv
| Column | Type | Description |
|--------|------|-------------|
| Accession | str | Sequence accession |
| Year | int | Collection year |
| historical_cluster | str | Rule-based historical cluster |
| kmeans_cluster | int | K-Means computed cluster ID |

## phase3_variations_detected.csv
| Column | Type | Description |
|--------|------|-------------|
| accession | str | Sequence accession |
| subtype | str | H1N1 or H3N2 |
| year | int | Collection year |
| position | int | Zero-based position in alignment |
| ref_char | str | Character in reference sequence |
| var_char | str | Character in query sequence |

## phase3_variations_annotated.csv
Extends phase3_variations_detected.csv with:
| in_critical_region | bool | Position falls in defined antigenic site |
| in_binding_region | bool | Position falls in receptor binding site |

## phase3_top_variations.csv
| Column | Type | Description |
|--------|------|-------------|
| position | int | Alignment position |
| subtype | str | H1N1 or H3N2 |
| ref_char | str | Reference amino acid |
| var_char | str | Variant amino acid |
| frequency | int | Count across all sequences |

## phase4_representative_strains.csv
Subset of full data with columns from phase1_h1n1_filtered.csv.

## phase4_distance_matrix_h1n1.csv / phase4_distance_matrix_h3n2.csv
Square NxN matrix; rows and columns indexed by Accession.
Values are normalised Hamming distances (0-1).

## phase4_mds_coordinates.csv
| Column | Type | Description |
|--------|------|-------------|
| Accession | str | Sequence accession |
| Year | int | Collection year |
| Subtype | str | H1N1 or H3N2 |
| mds_x | float | MDS dimension 1 coordinate |
| mds_y | float | MDS dimension 2 coordinate |
| historical_cluster | str | Cluster name (H3N2 only; N/A for H1N1) |

## phase5_variant_tracking.csv
| Column | Type | Description |
|--------|------|-------------|
| year | int | Year (2009-2020) |
| n_sequences | int | Total H1N1 sequences in that year |
| n_critical_variations | int | Variations at critical sites |
| pct_with_variation | float | % of sequences with critical variation |
