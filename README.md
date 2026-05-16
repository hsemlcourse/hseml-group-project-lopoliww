# Прогнозирование спроса на онлайн-курсы Stepik

**Тип задачи:** регрессия\
**Цель:** предсказать количество учащихся (`learners_count`) по данным о курсах, уроках и шагах онлайн-курсов.\
**Источник данных:** данные собраны самостоятельно через Stepik API.\
**Основная метрика:** `RMSE` на логарифмированной цели `log1p(learners_count)`.

Проект посвящён оценке потенциального спроса на образовательный контент. Модель может использоваться как вспомогательный инструмент для анализа курсов: какие характеристики курса, уроков и шагов связаны с большим числом учащихся.

------------------------------------------------------------------------

## Содержание

1.  [Описание задачи](#описание-задачи)
2.  [Данные](#данные)
3.  [Структура проекта](#структура-проекта)
4.  [Быстрый старт](#быстрый-старт)
5.  [Запуск EDA](#запуск-eda)
6.  [Обучение моделей](#обучение-моделей)
7.  [Результаты](#результаты)
8.  [Интерпретация модели](#интерпретация-модели)
9.  [Проверка качества кода](#проверка-качества-кода)
10. [Docker](#docker)
11. [Артефакты](#артефакты)

------------------------------------------------------------------------

## Описание задачи {#описание-задачи}

В проекте решается задача регрессии: по признакам курса, урока и шага нужно предсказать количество учащихся на курсе.

Целевая переменная `learners_count` имеет сильно скошенное распределение: у большинства курсов небольшое число учащихся, но есть отдельные очень популярные курсы. Поэтому для обучения и основной оценки качества используется преобразование:

``` python
log_target = log1p(learners_count)
```

Основная метрика — `RMSE` на логарифмированной цели. Она выбрана как основная, потому что сильнее штрафует крупные ошибки, чем `MAE`, и это важно для задачи прогнозирования спроса: недооценка или переоценка крупных курсов особенно заметна. Дополнительно считаются `MAE`, `R²`, а также `RMSE` и `MAE` после обратного преобразования в исходную шкалу учащихся.

Если метрики расходятся, приоритет отдаётся `RMSE_log`, потому что именно она отражает целевую оптимизацию модели и чувствительна к большим ошибкам. `MAE_log` используется как дополнительная проверка устойчивости, а `R²_log` — как показатель объяснённой дисперсии.

------------------------------------------------------------------------

## Данные {#данные}

Используемый файл после обработки:

``` text
data/processed/stepik_course_steps_processed.csv
```

Фактический размер датасета после обработки:

``` text
16 289 строк × 53 столбца
```

Разбиение:

| Split | Количество строк |
|-------|-----------------:|
| train |           12 942 |
| val   |            1 698 |
| test  |            1 649 |

Разбиение сделано заранее и хранится в колонке `split`. Для уменьшения риска leakage используется группировка по курсам: строки одного курса не должны одновременно попадать в train и validation/test.

------------------------------------------------------------------------

## Структура проекта {#структура-проекта}

``` text
.
├── data
│   ├── raw                         # исходные данные
│   └── processed                   # обработанные данные
├── models                          # сохранённые модели и метрики
│   ├── best_model.joblib
│   ├── metrics.csv
│   ├── metadata.json
│   ├── outlier_report.csv
│   ├── permutation_importance.csv
│   ├── hist_gradient_boosting_random_search_cv_results.csv
│   └── extra_trees_random_search_cv_results.csv
├── notebooks                       # ноутбуки EDA / baseline / experiments, если используются
├── report
│   ├── images                      # графики и таблицы EDA
│   └── report.md                   # основной отчёт
├── src
│   ├── preprocessing.py            # сборка и предобработка данных
│   ├── eda_plots.py                # EDA-графики и анализ выбросов
│   └── modeling.py                 # обучение, подбор гиперпараметров, оценка
├── tests
│   ├── test.py
│   └── test_cp2_additions.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## Быстрый старт {#быстрый-старт}

Команды для PowerShell из корня репозитория:

``` powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Проверка наличия данных:

``` powershell
Test-Path "data\processed\stepik_course_steps_processed.csv"
python -c "import pandas as pd; df=pd.read_csv('data/processed/stepik_course_steps_processed.csv'); print(df.shape); print(df['split'].value_counts())"
```

Ожидаемый результат:

``` text
(16289, 53)
train    12942
val       1698
test      1649
```

------------------------------------------------------------------------

## Запуск EDA {#запуск-eda}

``` powershell
python -m src.eda_plots `
  --input data/processed/stepik_course_steps_processed.csv `
  --output-dir report/images
```

После запуска создаются:

``` text
report/images/target_distribution_raw.png
report/images/target_distribution_log.png
report/images/eda_outlier_summary.csv
report/images/feature_target_correlations.csv
report/images/feature_vs_target_*.png
```

В EDA добавлены:

-   анализ распределения таргета до и после логарифмирования;
-   анализ выбросов по числовым признакам;
-   визуализации зависимостей признаков от таргета;
-   таблица корреляций числовых признаков с `log1p(learners_count)`.

------------------------------------------------------------------------

## Обучение моделей {#обучение-моделей}

Быстрый запуск без подбора гиперпараметров:

``` powershell
python -m src.modeling `
  --input data/processed/stepik_course_steps_processed.csv `
  --output-dir models `
  --no-search
```

Полный запуск с подбором гиперпараметров:

``` powershell
python -m src.modeling `
  --input data/processed/stepik_course_steps_processed.csv `
  --output-dir models `
  --search-iter 12
```

В полном запуске используются:

-   `DummyRegressor` как baseline;
-   `Ridge`;
-   `RandomForestRegressor`;
-   `ExtraTreesRegressor`;
-   `HistGradientBoostingRegressor`;
-   `RandomizedSearchCV` для `HistGradientBoostingRegressor` и `ExtraTreesRegressor`.

------------------------------------------------------------------------

## Результаты {#результаты}

Фактические результаты последнего полного запуска с `--search-iter 12`:

| Модель | Split | RMSE log | MAE log | R² log | RMSE learners | MAE learners | Комментарий |
|----|----|---:|---:|---:|---:|---:|----|
| Dummy median | val | 1.7362 | 1.6075 | -0.6052 | 30.5613 | 19.0901 | простая baseline-модель |
| Ridge alpha=1 | val | 1.4898 | 1.1730 | -0.1819 | 34.6265 | 17.6735 | линейная модель |
| Ridge alpha=10 | val | 1.4849 | 1.1695 | -0.1742 | 34.4860 | 17.6088 | линейная модель |
| ExtraTrees default | val | 1.0812 | 0.8723 | 0.3775 | 28.0479 | 13.1330 | ансамбль деревьев |
| RandomForest default | val | 1.0702 | 0.8461 | 0.3900 | 27.3066 | 12.8845 | ансамбль деревьев |
| ExtraTrees random search | val | 1.0601 | 0.8709 | 0.4016 | 27.6173 | 12.9220 | подбор гиперпараметров |
| HistGradientBoosting default | val | 1.0370 | 0.7751 | 0.4273 | 29.1472 | 12.9945 | градиентный бустинг |
| HistGradientBoosting random search | val | **0.9924** | **0.7296** | **0.4755** | 28.2435 | **12.4954** | лучшая validation-модель |
| HistGradientBoosting random search | test | **1.3758** | **0.9644** | **0.4665** | 170.2145 | 53.3544 | финальная test-оценка |

Лучшая модель по основной метрике `RMSE_log` на validation — `HistGradientBoostingRegressor` после `RandomizedSearchCV`.

Лучшие найденные гиперпараметры:

``` json
{
  "model__l2_regularization": 1.0,
  "model__learning_rate": 0.08,
  "model__max_iter": 350,
  "model__max_leaf_nodes": 31,
  "model__min_samples_leaf": 20
}
```

------------------------------------------------------------------------

## Интерпретация модели {#интерпретация-модели}

Для финальной модели рассчитана permutation importance. Самые важные признаки:

| Признак                  | Увеличение RMSE при перестановке |
|--------------------------|---------------------------------:|
| `time_to_complete`       |                           0.6964 |
| `_parsed_page`           |                           0.1830 |
| `lessons_count`          |                           0.1806 |
| `title_len`              |                           0.1761 |
| `price_numeric`          |                           0.0563 |
| `title_word_count`       |                           0.0526 |
| `is_paid`                |                           0.0397 |
| `summary_len`            |                           0.0278 |
| `description_word_count` |                           0.0268 |
| `lesson_steps_count`     |                           0.0238 |

Вывод: сильнее всего на прогноз влияют признаки, связанные с длительностью прохождения, объёмом курса, текстовым описанием и коммерческими характеристиками курса.

------------------------------------------------------------------------

## Проверка качества кода {#проверка-качества-кода}

Последний подтверждённый локальный запуск:

``` powershell
ruff check src tests
```

Результат:

``` text
All checks passed!
```

Тесты:

``` powershell
pytest tests/test.py tests/test_cp2_additions.py -q
```

Результат:

``` text
9 passed, 15 warnings
```

Предупреждения относятся к будущему изменению поведения `pandas` при `fillna` и не ломают текущую работу пайплайна.

------------------------------------------------------------------------

## Docker {#docker}

В проект добавлены:

``` text
Dockerfile
docker-compose.yml
.dockerignore
```

Запуск полного пайплайна через Docker:

``` powershell
docker compose up --build
```

Остановка и удаление контейнера:

``` powershell
docker compose down
```

Docker-пайплайн предназначен для воспроизводимого запуска EDA, обучения моделей, тестов и линтера в контейнере. Финальный подтверждённый Docker-прогон завершился успешно: `pytest` внутри контейнера показал `9 passed`, `ruff` показал `All checks passed!`, контейнер `stepik-demand-cp2` завершился с `code 0`.

------------------------------------------------------------------------

## Артефакты {#артефакты}

Основные артефакты после запуска:

``` text
models/best_model.joblib
models/metrics.csv
models/metadata.json
models/outlier_report.csv
models/permutation_importance.csv
models/hist_gradient_boosting_random_search_cv_results.csv
models/extra_trees_random_search_cv_results.csv
report/images/eda_outlier_summary.csv
report/images/feature_target_correlations.csv
report/images/feature_vs_target_*.png
```

------------------------------------------------------------------------

## Краткий вывод

По сравнению с baseline качество существенно улучшилось: `RMSE_log` снизился с `1.7362` у `DummyRegressor` до `0.9924` у финальной модели на validation. Подбор гиперпараметров улучшил `HistGradientBoostingRegressor` относительно дефолтной версии. Финальная test-оценка хуже validation, что указывает на различие распределений между validation и test или наличие более сложных примеров в test-наборе, но модель всё равно заметно лучше baseline и объясняет часть дисперсии таргета (`R²_log = 0.4665` на test).
