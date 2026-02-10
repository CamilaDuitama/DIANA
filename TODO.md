# Project TODOs
- [ ] 15. Add input file validation in diana-predict to detect insufficient data:
  - Check minimum sequence count (current: only checks if file size > 0 bytes)
  - Use `seqkit stats` to verify files have enough reads (e.g., >10,000 sequences)
  - Detect files with very short reads or no k-mer matches before running full pipeline
  - Currently files with ~4,000-5,000 sequences (e.g., SRR867035, SRR867040) fail late in pipeline with "Loaded 0 unitig fractions" error
  - Should fail early with clear message: "Insufficient sequences (4,088 < 10,000 minimum)"
- [x] 29. Verify the data split its still ok with the samples 
- [ ] 30. Add the feature of diana project new samples inside scripts/paper/07_project_new_samples.py. THe idea is the user can do diana-project and then see where they sample falls in a PCA of unitigs and a PCA of samples.
- [ ] 31. There is one sample whose sample host is homo sapiens oral and i think this is a mistake. It should be homo sapiens. I think it is in the validation dataset. Correct this and confirm.





