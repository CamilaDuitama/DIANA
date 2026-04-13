# External Tools

This directory contains source code for third-party tools compiled locally or used as submodules.

## Tools

### 1. muset / kmat_tools
**Purpose**: K-mer matrix generation from sequencing data

**Installation**: Installed automatically via conda (see `environment.yml`):
```bash
mamba install -c camiladuitama muset
```

No manual build required. `install.sh` verifies availability.

**Repository**: https://github.com/rvicedomini/muset

---

### 2. back_to_sequences
**Purpose**: Map k-mers back to the original reads that contain them

**Installation**: Built automatically by `install.sh` from `external/back_to_sequences/`:
```bash
bash install.sh
```

This compiles a Rust binary and installs it to the active conda environment's `bin/`.

**Repository**: https://github.com/pierrepeterlongo/back_to_sequences

---

## Notes

- `back_to_sequences/` is a git submodule (see `.gitmodules`)
- `muset/` is excluded from git via `.gitignore`
- All installation is handled by `install.sh` — no manual steps required
