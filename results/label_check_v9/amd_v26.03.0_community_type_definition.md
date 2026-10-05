# AncientMetagenomeDir v26.03.0: how community_type and material are defined

Source: AncientMetagenomeDir git tag v26.03.0 (commit 36f953aa, 2026-04-17), https://github.com/SPAAM-community/AncientMetagenomeDir.
Our tables (data/metadata/AncientMetagenomeDir-v26.03.0/) match that tag byte for byte (md5 4bc9bb0f... samples, 25e76844... libraries).

## assets/documentation/samples/README.md (the samples README links to it), sections community_type and material

```
## community_type

> 🧫 host-associated metagenome only!

- The type of community from the host's original body the sample is derived from.
  - e.g. oral, gut

> ⚠️ Must follow categories specified in `assets/enums/<column>.json`


## material

- Sample type DNA was extracted from
  - e.g. dental calculus, palaeofaeces, intestinal, chewing gum
  - e.g. permafrost, lake sediment, peat soil, bone
  - e.g. tooth, bone, dental calculus

- For host-associated single genome list only:
  - If genome is derived from multiple tissue types from the same individual (e.g. bone and soft tissue) then the entry should simply be listed as 'tissue'

> ⚠️ Partly [MIxS v5](https://gensc.org/mixs/) compliant field, i.e. term
> from an [ontology](https://www.ebi.ac.uk/ols/index), and ideally either
> [UBERON](https://www.ebi.ac.uk/ols/ontologies/uberon) (anatomy) or
> [ENVO](https://www.ebi.ac.uk/ols/ontologies/envo) (everything else). If you
> can't find something close enough, please ping
> @spaam-community/ancientmetagenomedir-coreteam

> ⚠️ Must follow categories specified in `assets/enums/<column>.json`

> ⚠️ Mandatory value

```

## Schema (ancientmetagenome-hostassociated_samples_schema.json)

- community_type: title "Type of the host's community sample represents", description "The type of the original host's community the sample contains", values from assets/enums/community_type.json
- material: title "Type of material the host's community was selected from", description "Sample type DNA was extracted from", values from assets/enums/material.json

## Allowed community_type values at v26.03.0

```
{
    "enum": [
        "oral",
        "gut",
        "soft tissue",
        "skeletal tissue",
        "plant tissue",
        "leaf",
        "root"
    ]
}
```

## CHANGELOG lines re-listing tooth samples (release heading in brackets)

```
[## v20.09: Ancient Ksour of Ouadane] - Willmann2018: re-list the community_type of tooth samples as skeletal tissue
[## v20.09: Ancient Ksour of Ouadane] - Brealey2020: re-list the community_type of tooth samples as skeletal tissue
```
