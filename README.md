# Abalone Age Prediction

Machine learning project to predict abalone age from physical measurements using multiple algorithms.

## Dataset
- **Source**: UCI Machine Learning Repository
- **Features**: Sex, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight
- **Target**: Rings (age indicator)
- **Samples**: 4,177

## Models Compared
- Decision Tree
- Random Forest  
- XGBoost
- Gradient Boosting
- MLP (Adam)
- MLP (SGD)

## Key Results
| Model | Best Params | Accuracy | F1-Score |
|-------|-------------|----------|----------|
| XGBoost | n_estimators=300 | 0.83 | 0.83 |
| Random Forest | n_estimators=1200 | 0.81 | 0.81 |
| Decision Tree | max_depth=5 | 0.72 | 0.72 |

## Key Insights
- Shell weight has highest correlation with age (r=0.627)
- Diameter is second most important feature (r=0.574)
- Age groups: 1 (0-7), 2 (8-10), 3 (11-15), 4 (16+ rings)
