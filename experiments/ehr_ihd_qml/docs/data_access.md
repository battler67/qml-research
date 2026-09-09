# Official data access

## UCI

The runner uses UCI dataset 45 through the repository's authoritative `ucimlrepo` loader and cache.
Downloaded data is ignored by Git; a provenance manifest and SHA-256 checksum are recorded.

## Framingham

Request or download the anonymized teaching dataset from
<https://biolincc.nhlbi.nih.gov/teaching/> and verify its documentation at
<https://biolincc.nhlbi.nih.gov/media/teachingstudies/FHS_Teaching_Longitudinal_Data_Documentation_2021a.pdf>.
Place the verified CSV at:

```text
data/raw/framingham/fhs_longitudinal.csv
```

Do not substitute an arbitrary `framingham.csv`. Record the source page, retrieval date, license or
terms, checksum, row count, and column names. The teaching data is anonymized and suitable for
instruction and pipeline validation, not publication-quality clinical claims.
