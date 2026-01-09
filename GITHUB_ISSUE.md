# [ENHANCEMENT] Increase Training Dataset Size

## Problem Statement

The current training dataset (`data/train.csv`) contains only **81 samples** (approximately 40 positive and 41 negative examples), which is insufficient for training a robust LSTM-based sentiment analysis model. This limited dataset size can lead to:

- Poor generalization to unseen text
- Overfitting on the training data
- Reduced accuracy on real-world examples
- Limited vocabulary coverage

## Current Dataset Statistics

- **Total samples**: 81
- **Positive samples**: ~40
- **Negative samples**: ~41
- **Format**: CSV with columns `text` and `sentiment` (0 for negative, 1 for positive)
- **Average text length**: Very short (mostly single words or short phrases)

## Proposed Solution

Expand the training dataset to include:

1. **Minimum target**: 1,000+ samples per class (2,000+ total)
2. **Ideal target**: 5,000+ samples per class (10,000+ total)
3. **Diversity improvements**:
   - Longer sentences and phrases
   - Various domains (reviews, social media, news, etc.)
   - Different writing styles and contexts
   - More nuanced sentiment expressions

## Dataset Sources (Suggestions)

- **Public datasets**:
  - IMDb Movie Reviews
  - Amazon Product Reviews
  - Twitter Sentiment Analysis datasets
  - Yelp Reviews
  - Rotten Tomatoes reviews

- **Data augmentation**:
  - Synonym replacement
  - Paraphrasing
  - Back-translation

## Expected Benefits

- ✅ Improved model accuracy and generalization
- ✅ Better handling of diverse text inputs
- ✅ Reduced overfitting
- ✅ More robust predictions on real-world data
- ✅ Enhanced vocabulary coverage

## Implementation Notes

- Dataset should maintain the same CSV format: `text,sentiment`
- Ensure balanced distribution between positive (1) and negative (0) classes
- Maintain data quality (remove duplicates, handle noise)
- Update `src/train.py` if needed to handle larger datasets efficiently
- Consider updating `TextPreprocessor` parameters (`max_vocab_size`, `max_seq_length`) for larger vocabulary

## Acceptance Criteria

- [ ] Dataset expanded to at least 1,000 samples per class
- [ ] Dataset maintains balanced class distribution
- [ ] Dataset format remains compatible with existing code
- [ ] Model retrained with new dataset shows improved performance
- [ ] Validation accuracy improves significantly
- [ ] Model generalizes better to test examples

## Additional Context

The current model architecture in `src/model.py` and training script in `src/train.py` should be able to handle a larger dataset without major modifications, though training time will increase.

---

**Priority**: High  
**Effort**: Medium  
**Impact**: High

