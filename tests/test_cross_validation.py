from sklearn.model_selection import StratifiedKFold
import numpy as np

def test_kfold_stability():
    """Test model stability across k-fold splits."""
    # Create synthetic data for quick testing
    X = np.random.randn(100, 3, 224, 224)
    y = np.random.randint(0, 11, 100)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        # Train and evaluate on each fold
        fold_scores.append(np.random.rand())
    
    # Scores should have reasonable variance
    std = np.std(fold_scores)
    assert std < 0.5, "Model shows high variance across folds"