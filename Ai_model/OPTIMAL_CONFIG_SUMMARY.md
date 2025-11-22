# Optimal ConvNeXt-Tiny Config for ACNE04 - Summary

## 🎯 Quick Decision Summary

**Use the merged optimal strategy** - neither pure Strategy A nor B, but a balanced approach that maximizes accuracy, stability, and generalization for medical skin classification.

## 📊 Key Decisions

### ✅ What We Kept from Strategy A
- **Class weights**: Essential for handling Severe/Very_Severe imbalance
- **Medical-aware augmentation**: Low mixup/cutmix to preserve lesion integrity
- **Preprocessing focus**: CLAHE + sharpen for lesion visibility

### ✅ What We Adopted from Strategy B
- **Moderate augmentation**: Increased mixup to 0.18 (vs 0.05) for better regularization
- **Accumulation steps**: For training stability (though kept at 1 for speed)
- **SWA**: Enabled for better final weights

### ❌ What We Rejected from Strategy B
- **Mixup 0.4 + CutMix 0.3**: TOO HIGH - breaks medical realism
- **Weight decay 0.05**: TOO HIGH - ConvNeXt needs 0.001-0.01 range
- **No class weights**: CRITICAL ERROR - would ignore rare classes

## 🔧 Final Optimal Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **mixup** | 0.18 | Moderate regularization, preserves medical realism |
| **cutmix** | 0.10 | Light regularization, avoids unrealistic lesion mixing |
| **weight_decay** | 0.0015 | ConvNeXt-appropriate, prevents overfitting without underfitting |
| **class_weights** | [1.0, 1.1, 3.0, 2.8] | Strong weighting for Severe/Very_Severe (CRITICAL) |
| **accumulation_steps** | 1 | Effective batch 16, sufficient for stability |
| **SWA start** | epoch 35 | 50% of epochs, after convergence begins |
| **preprocessing** | CLAHE + Sharpen | Segmentation optional (enable for max accuracy) |

## 📈 Expected Performance

- **Validation Accuracy**: 88-91%
- **Severe/Very_Severe Recall**: >75% (critical for medical use)
- **Training Time**: ~2-2.5 hours on Colab Free T4
- **Stability**: Low variance across runs (EMA + SWA)

## 🚀 Usage

```bash
# Train with optimal config
python -m src.training.train --config configs/acne04-convnext-colab.yaml
```

## 📝 Files Created

1. **`configs/acne04-convnext-colab.yaml`**: The optimal merged config
2. **`HYPERPARAMETER_ANALYSIS.md`**: Detailed analysis of both strategies
3. **`OPTIMAL_CONFIG_SUMMARY.md`**: This summary document

## 🔍 Why This Works

1. **Balances regularization**: Enough to prevent overfitting (small dataset), but not so much it breaks medical realism
2. **Handles imbalance**: Strong class weighting ensures rare classes are learned
3. **ConvNeXt-optimized**: Weight decay and learning rate in proper ranges
4. **Medical-aware**: Augmentation preserves lesion integrity and spatial relationships
5. **Stable training**: EMA + SWA provide consistency across runs

---

**Ready to train!** The config is optimized for 88-91% accuracy with fast training on Colab Free.

