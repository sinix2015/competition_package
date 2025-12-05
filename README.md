# Full Training Pipeline

1. Установите окружение Python 3.12 с PyTorch и scikit-learn (см. `requirements` базового окружения).
2. Запустите `python training/train_models.py` из корня проекта. Скрипт поддерживает JSON-конфиг:
   ```bash
   python training/train_models.py --config-path configs/your_best.json --atrifacts-dir atrifacts/your_best_model
   ```
   По умолчанию выполняется 5-fold CV и финальное обучение;
   ключи `--skip-final`, `--no-save`, `--quiet` позволяют быстро проверить конфигурацию без перезаписи артефактов.
   Скрипт:
   - Строит онлайн-признаки с помощью `src.features.FeatureGenerator`. Так же есть и `minimal_features, advanced_features, simple_features` и экспериментальный DLinearAugmentedFeatureGenerator `experiments.dlinear.feature_generator`.
   - Делит последовательности по `n_folds` (по умолчанию 5).
   - Обучает ансамбль из MLP, Ridge и Transformer.
   - Сохраняет веса и статистики в `artifacts/` (при включённом финальном обучении).
   - Возвращает JSON-резюме с OOF и финальными метриками.
3. После обучения `solution.py` автоматически подхватывает новые веса.

Артефакты:
- `data_stats.npz` — нормировочные статистики.
- `feature_mlp.pt`, `ridge_weights.npz`, `temporal_transformer.pt` — веса базовых моделей.
- `ensemble_weights.npz` — коэффициенты линейного стекера.
- `training_summary.json` — контрольные метрики (R² на валидации).
- `auto_training_summary.json` — метрики финального запуска при использовании автоматического тюнера.
