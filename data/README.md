# Dataset provenance

Only public, non-identifiable datasets are used.

| Dataset | Loader/source | Phase 1 use | Positive class |
| --- | --- | --- | --- |
| Binary Iris | `sklearn.datasets.load_iris` | Tutorial sanity check only | Iris class 1 |
| WDBC | `sklearn.datasets.load_breast_cancer` | Primary diagnostic benchmark | Malignant |
| Cleveland Heart Disease | UCI dataset 45 via `ucimlrepo` | Cross-disease benchmark | `num > 0` |

The Heart Disease loader caches a normalized CSV and a JSON provenance manifest
under `data/raw/`. The directory is ignored because the authoritative public source
can be fetched again; the source DOI, retrieval time, shape, and SHA-256 checksum are
included in every experiment record.

WDBC describes features computed from digitized breast-mass fine-needle aspirate
images. It must not be described as pre-symptomatic or longitudinal screening data.

