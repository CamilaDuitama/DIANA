# External Tools

This directory contains third-party tools required for DIANA. These tools are not included in the repository and must be installed separately.

## Required Tools

### 1. Muset
**Purpose**: K-mer matrix generation from sequencing data

**Installation**:
```bash
cd external/
git clone https://github.com/tlemane/muset.git
cd muset
# Follow installation instructions in muset/README.md
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j 4
```

**Repository**: https://github.com/tlemane/muset

---

### 2. SeqDD (optional)
**Purpose**: Fast sequence data download

**Installation**:
```bash
cd external/
git clone https://github.com/rki-mf1/SeqDD.git seqdd
cd seqdd
# Follow installation instructions
```

**Repository**: https://github.com/rki-mf1/SeqDD

---

### 3. back_to_sequences (optional)
**Purpose**: Convert k-mer matrices back to sequences

**Installation**:
```bash
cd external/
git clone <repository-url> back_to_sequences
cd back_to_sequences
# Follow installation instructions
```

---

## Automated Installation

You can use the provided installation scripts:

```bash
# Install muset
bash scripts/build_muset.sh

# Install seqdd (if needed)
bash scripts/install_seqdd.sh
```

---

## Directory Structure After Installation

```
external/
├── README.md           # This file
├── muset/             # Muset tool (git ignored)
│   └── bin/
│       ├── muset
│       └── kmat_tools
├── seqdd/             # SeqDD tool (git ignored)
└── back_to_sequences/ # back_to_sequences tool (git ignored)
```

---

## Notes

- These tools are excluded from git via `.gitignore`
- Each tool maintains its own git repository
- Installation scripts are provided in `scripts/` directory
- Refer to each tool's documentation for detailed installation instructions
