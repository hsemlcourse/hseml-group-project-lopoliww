# CP2-доработки по фидбеку

## 1. Метрика качества и правило выбора модели

Основная метрика проекта — `RMSE` по `log1p(learners_count)`. Я выбираю её как главную, потому что таргет — количество учеников — имеет длинный правый хвост: большая часть курсов маленькие, но у небольшой доли курсов число учеников сильно выше медианы. В такой задаче важно сильнее штрафовать крупные ошибки на популярных курсах: ошибка в несколько тысяч учеников бизнесово важнее ошибки в несколько десятков учеников. Поэтому RMSE лучше подходит как основная метрика, чем MAE: MAE показывает среднюю абсолютную ошибку и устойчивее к выбросам, но слабее реагирует на редкие крупные промахи.

`R²` используется как дополнительная диагностическая метрика: она показывает, насколько модель лучше константного прогноза объясняет вариативность таргета. При выборе модели я не оптимизирую напрямую `R²`, потому что в прикладной задаче важнее абсолютный размер ошибки прогноза. Если метрики расходятся, правило такое: сначала выбирается модель с минимальным validation `RMSE log`; затем проверяется, что у неё нет сильного проигрыша по `MAE log` и `R² log`. Если две модели имеют близкий `RMSE log`, предпочтение отдаётся более простой и стабильной модели с лучшим `MAE log` и более интерпретируемыми признаками.

## 2. Анализ и обработка выбросов

Выбросы проверяются в двух местах:

1.  В EDA-скрипте `src/eda_plots.py` создаётся таблица `report/images/eda_outlier_summary.csv`. Для каждого числового признака сохраняются `min`, `p01`, `median`, `p99`, `max`, число и доля IQR-выбросов.
2.  В обучении `src/modeling.py` создаётся `models/outlier_report.csv` и используется `QuantileClipper`: числовые признаки клиппируются по 1% и 99% квантилям, причём границы считаются только на train/fold внутри sklearn pipeline. Это важно, чтобы не допустить утечку информации из validation/test.

Таргет `learners_count` не удаляется по верхним значениям, потому что очень популярные курсы — реальные наблюдения, а не ошибки парсинга. Вместо удаления применяется `log1p(learners_count)`: это сохраняет порядок курсов, но уменьшает влияние экстремально популярных курсов на обучение.

## 3. Визуализации зависимостей признаков от таргета

Добавлен скрипт:

``` bash
python -m src.eda_plots \
  --input data/processed/stepik_course_steps_processed.csv \
  --output-dir report/images
```

Он строит графики:

-   `target_distribution_raw.png` и `target_distribution_log.png` — распределение таргета до и после логарифмирования;
-   `feature_vs_target_rating.png` — связь рейтинга с `log1p(learners_count)`;
-   `feature_vs_target_reviews_count.png` — связь числа отзывов с таргетом;
-   `feature_vs_target_lessons_count.png`, `feature_vs_target_sections_count.png`, `feature_vs_target_lesson_steps_count.png` — связь сложности/структуры курса со спросом;
-   `feature_vs_target_language.png`, `feature_vs_target_step_type.png`, `feature_vs_target_is_paid.png`, `feature_vs_target_has_certificate.png` — средний таргет по категориям;
-   `feature_target_correlations.csv` — численная таблица корреляций признаков с лог-таргетом.

Выводы, которые нужно вставить после локального просмотра графиков:

-   признаки популярности курса (`reviews_count`, `rating`, `has_certificate`) ожидаемо связаны с числом учеников;
-   структурные признаки (`lessons_count`, `sections_count`, `lesson_steps_count`) дают сигнал о масштабе курса, но зависимость не строго линейная;
-   категориальные признаки (`language`, `step_type`, `is_paid`) полезны, потому что разные типы курсов и шагов имеют разный средний спрос;
-   из-за нелинейных зависимостей деревья и градиентный бустинг выглядят более подходящими, чем чисто линейная Ridge-модель.

## 4. Перебор гиперпараметров

В `src/modeling.py` добавлен `RandomizedSearchCV` для двух сильных семейств моделей:

-   `HistGradientBoostingRegressor`: подбираются `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`;
-   `ExtraTreesRegressor`: подбираются `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.

Для кросс-валидации используется `GroupKFold` по `course_id`, чтобы шаги одного курса не попадали одновременно в train и validation fold. Это продолжает идею group split и защищает от dataleak.

Запуск:

``` bash
python -m src.modeling \
  --input data/processed/stepik_course_steps_processed.csv \
  --output-dir models \
  --search-iter 12
```

Результаты сохраняются в:

-   `models/metrics.csv` — итоговая таблица моделей;
-   `models/hist_gradient_boosting_random_search_cv_results.csv`;
-   `models/extra_trees_random_search_cv_results.csv`;
-   `models/best_model.joblib`;
-   `models/metadata.json`.

## 5. Финальная модель и интерпретируемость

Финальная модель выбирается по validation `RMSE log`, но выбор теперь обосновывается не только одной цифрой. Я сравниваю:

1.  качество относительно `DummyRegressor` — показывает, что модель действительно учит закономерности;
2.  качество относительно `Ridge` — проверяет, достаточно ли линейной зависимости;
3.  качество деревьев и бустинга — проверяет нелинейные зависимости и взаимодействия признаков;
4.  `MAE log` и `R² log` — проверка, что выигрыш по RMSE не достигнут ценой нестабильности;
5.  важность признаков — `models/permutation_importance.csv`.

Интерпретация считается через `permutation_importance` на validation split. Это модельно-независимый способ: признак считается важным, если перемешивание его значений увеличивает ошибку. В отчёте нужно привести топ-10 признаков из `models/permutation_importance.csv` и коротко объяснить, почему они логичны для спроса на онлайн-курс.

## 6. Docker / docker-compose

Добавлены:

-   `Dockerfile` — собирает окружение, устанавливает зависимости и запускает EDA + обучение;
-   `docker-compose.yml` — запускает полный pipeline: EDA, обучение с search, тесты и ruff;
-   `.dockerignore` — исключает кэш, виртуальные окружения и тяжёлые служебные файлы.

Проверка:

``` bash
docker compose up --build
```

После запуска должны обновиться `report/images/*`, `models/metrics.csv`, `models/permutation_importance.csv`, `models/outlier_report.csv` и `models/best_model.joblib`.

## 7. Что именно закрывает фидбек

| Фидбек | Что добавлено |
|----|----|
| Нет пояснения, почему RMSE основной | Добавлено правило выбора: RMSE log основной, MAE/R² диагностические |
| Нет анализа и обработки выбросов | `eda_outlier_summary.csv`, `outlier_report.csv`, `QuantileClipper` внутри pipeline |
| Нет зависимостей фичей от таргета | `src/eda_plots.py` и графики `feature_vs_target_*.png` |
| Выбор модели только по RMSE | Добавлено сравнение с baseline/Ridge/ансамблями + permutation importance |
| Нет перебора гиперпараметров | `RandomizedSearchCV` + `GroupKFold` + сохранение cv-results |
| Нет docker/docker-compose | Добавлены `Dockerfile`, `docker-compose.yml`, `.dockerignore` |
