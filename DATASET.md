# Dataset Notes

This repository does not distribute the original imaging data.

## Data Access

Users must obtain the dataset through the appropriate official access process and comply with all data use agreements, IRB rules, and institutional requirements.

## Expected Files

The code expects at least:
- a dataset JSON file, for example `ISPY2_json_minimal.json`
- a split JSON file, for example `split_seed20.json`

## Expected Dataset JSON Fields

Each patient entry is expected to contain fields similar to:

```json
{
  "PATIENT_ID": {
    "image_0": "/path/to/pre_contrast.nii.gz",
    "image_1": "/path/to/post_contrast.nii.gz",
    "mask": "/path/to/mask.nii.gz",
    "pcr": 0,
    "clinical_vec": [0.0, 1.0, 0.0],
    "clinical_feature_names": ["feature_a", "feature_b", "feature_c"]
  }
}
```

Notes:
- `image_0` and `image_1` are required by the current preprocessing logic.
- `mask` is optional. If it is missing, the loader falls back to an empty mask.
- `clinical_vec` and `clinical_feature_names` are optional unless you use the clinical CBM options.

## Expected Split JSON Fields

The split file should contain:

```json
{
  "train_pids": ["PATIENT_1", "PATIENT_2"],
  "val_pids": ["PATIENT_3"],
  "test_pids": ["PATIENT_4"]
}
```

The code also tolerates `train_ids`, `val_ids`, and `test_ids` as fallback field names in some places.

## Concept Embeddings

The CBM script also requires a concept embedding file in `.pt` format:

```text
concept_embeddings.pt
```

This repository does not include the embedding file. It should be prepared separately.

## Privacy

Before publishing any derived metadata files, make sure they do not contain:
- private directory paths
- internal usernames
- unreleased identifiers
- patient-sensitive information beyond what is allowed by your data agreement
