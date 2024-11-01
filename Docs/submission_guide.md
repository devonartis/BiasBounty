# Bias Bounty 2 - Submission Guide

## Required Files for Submission

1. **Model File**
   - `model/kmeans_model.pkl` (trained model)

2. **Scripts**
   - `2.inference.py` (main inference script)
   - `utils.py` (utilities and model class)
   - `config.py` (path configurations)

3. **Prediction Files**
   - `predictions.csv` (test dataset predictions)
   - `adversarial_predictions.csv` (perturbed images predictions)

4. **Advanced Submission**
   - `perturbed/` directory with adversarial examples (.jpg format)

5. **Documentation**
   - `README.md` (setup and usage instructions)
   - `environment.yml` (dependencies)

## Submission Preparation Steps

1. **Clean Repository**
```bash
# Remove unnecessary files
rm -rf __pycache__/
rm -rf Training_data/
rm -rf Test_data/
rm 1.train.py
rm 3.adversarial.py
rm 4.advance_adversary.py
rm 5.visualize.py
```

2. **Verify Required Files**
```
submission/
├── model/
│   └── kmeans_model.pkl
├── perturbed/
│   └── [adversarial images].jpg
├── src/
│   ├── utils.py
│   ├── config.py
│   └── 2.inference.py
├── predictions.csv
├── adversarial_predictions.csv
├── environment.yml
└── README.md
```

3. **GitHub Setup**
```bash
# Create new private repository
git init
git add .gitignore README.md environment.yml
git add src/utils.py src/config.py src/2.inference.py
git add model/kmeans_model.pkl
git add perturbed/*.jpg
git add predictions.csv adversarial_predictions.csv
git commit -m "Initial submission"
git push -u origin main
```

4. **Add Collaborator**
- Add @NicoleScientist as collaborator

## Verification Steps

1. **Clone Fresh and Test**
```bash
# Clone repository
git clone [your-repo-url] test_submission
cd test_submission

# Create environment
conda env create -f environment.yml
conda activate hate-detection

# Run inference
python src/2.inference.py
```

2. **Check Outputs**
- Verify predictions.csv format:
  ```
  image_id,prediction_label
  [id1],0
  [id2],1
  ```
- Verify adversarial_predictions.csv format
- Check all .jpg files in perturbed/

## Submission Requirements Met

1. **Basic Requirements** ✓
   - Unsupervised model (K-means)
   - Binary predictions (0/1)
   - Model in .pkl format
   - Executable inference script

2. **Advanced Requirements** ✓
   - Adversarial examples (.jpg)
   - Multiple attack strategies
   - Perturbed predictions CSV

3. **Dependencies** ✓
   - environment.yml includes all requirements
   - Metal GPU optimization for Apple Silicon

## Pre-submission Checklist

- [x] Remove all training/test data
- [x] Verify model file exists and loads
- [x] Test inference script on fresh clone
- [x] Check all CSV formats
- [x] Verify all adversarial images are .jpg
- [ ] Add collaborator
- [x] Test environment creation
- [x] Update README with clear instructions

## Contact

For submission questions:
Devon Artis Troin.artis@gmail.com
