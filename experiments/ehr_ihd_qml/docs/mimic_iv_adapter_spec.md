# Future MIMIC-IV adapter specification

This task does not use MIMIC-IV as the primary incident-CHD cohort. A future adapter may identify
candidate coronary diagnoses using ICD-10 `I20-I25` and ICD-9 `410-414`, while retaining admission,
diagnosis, and discharge timestamps separately. It must define an index date, require observable
pre-index history, exclude prevalent disease, prevent discharge diagnoses and post-index tests from
entering predictors, and document out-of-network outcome blindness.

Controls must be sampled from an eligible at-risk population, not from a generic absence of a code.
MIMIC-IV-Ext Cardiac Disease is not a standalone binary dataset for this purpose because it mostly
contains positive cardiac cases and includes information from the completed diagnostic process.
