# Improved Artifacts - Anti-Overfitting MLP Training Results

This folder contains the results from the improved MLP classifier training with anti-overfitting techniques.

## Files

- `integrated_training_history.json` - Complete training metrics over 6 epochs
- `best_integrated_model.pt` - Best performing model checkpoint (if saved)

## Training Configuration

- **Model**: EnhancedClassifierWithFeatures (3-layer MLP)
- **Input**: 3,146 dimensions (embeddings + 74 code features)
- **Anti-overfitting**: Dropout 0.4, early stopping, LR scheduling
- **Batch size**: 16
- **Learning rate**: 1e-5 with ReduceLROnPlateau
- **Weight decay**: 1e-3 (L2 regularization)

## Results Summary

- **Early stopping**: Training stopped at epoch 6 (patience=5)
- **Stable metrics**: No degradation in code quality metrics
- **Final performance**: BERTScore=0.9047, CodeBLEU=0.4969 (stable)

## Key Improvements

✅ **Before**: Metrics degraded from epoch 1-19 (overfitting)
✅ **After**: Stable metrics with early stopping at epoch 6

This demonstrates the effectiveness of the anti-overfitting techniques in maintaining stable model performance.
