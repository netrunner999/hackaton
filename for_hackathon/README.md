### Meta Eval For Hackathon ###

## 🎯 Основная идея

Система реализует мета-оценку алгоритмов улучшения промптов:

1. **`Alg(D, E, M) = Prompt`** - Алгоритм улучшения промпта
   - Получает датасет D, конфигурацию evaluator E и модели M
   - Анализирует диалоги и улучшает промпт
   - Возвращает новый улучшенный промпт
   - **Примечание:** BaselineAlg выбирает 1 случайный диалог и улучшает промпт на его основе

2. **`metaeval(Alg, D, E, M)`** - Мета-оценка алгоритма
   - Запускает алгоритм улучшения: `Prompt = Alg(D, E, M)`
   - Оценивает полученный промпт через внешний Eval API: `evaluate(E, D, M, Prompt)`
   - Возвращает промпт и результаты оценки

**Важно:** Для работы системы необходим запущенный Eval API сервер (предоставляется организаторами).

## 📁 Структура проекта

```
for_hackathon/
├── src/
│   ├── pipelines/          # Пайплайны
│   │   ├── improve_algorithm.py       # BaselineAlg(D, E, M) = Prompt
│   │   ├── submission.py              # MyAlg(D, E, M) = Prompt (шаблон для пользовательских алгоритмов)
│   │   ├── selector.py                 # Alg(D, E, M) - выбор алгоритма (BaselineAlg или MyAlg)
│   │   └── meta_eval.py                # metaeval(Alg, D, E, M)
│   ├── core/               # Core infrastructure
│   │   ├── dataset.py      # Загрузка, валидация и подготовка датасетов
│   │   ├── runio.py        # Управление runs/<run_id>/
│   │   ├── llm.py          # Унифицированный LLM клиент (OpenAI + совместимые API)
│   │   ├── usage.py        # Логирование использования токенов
│   │   └── prompts.py      # Загрузка промптов
│   └── modules/            # Модули
│       └── regen.py        # Регенерация ответов агента
├── data/                   # Датасеты
│   ├── converted/          # Конвертированные датасеты
│   └── datasets_registry.json  # Реестр датасетов (ID -> путь)
└── runs/                   # Результаты запусков
    └── run_YYYYMMDD_HHMMSS/
        ├── config.json
        ├── prompts/
        ├── regen/
        ├── judge/
        ├── usage/
        └── final_report.json
```

## 🚀 Установка

### Требования

- Python 3.8+
- OpenAI-совместимый сервер (например, vLLM) или OpenAI API ключ

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Настройка

**Основной вариант: OpenAI-совместимый сервер (vLLM)**


В конфиге укажите `base_url`:
```json
{
  "models": {
    "model": {
      "name": "meta-llama/Llama-3.1-8B-Instruct",
      "base_url": "https://provider.intellemma.ru/v1",
      "temperature": 0.7,
      "max_tokens": 1000
    }
  }
}
```

Ключ задайте через переменную окружения `OPENAI_API_KEY`.

**Альтернатива: OpenAI API**

Если хотите использовать обычный OpenAI API, установите API ключ:
```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
```

Или создайте файл `.env`:
```
OPENAI_API_KEY=<YOUR_API_KEY>
```

В конфиге просто укажите модель без `base_url`:
```json
{
  "models": {
    "model": {
      "name": "gpt-4.1-mini",
      "temperature": 0.7,
      "max_tokens": 1000
    }
  }
}
```

## 📖 Использование

> 💡 **Быстрый старт:** Для быстрого начала работы см. файл [`QUICK_START.md`](QUICK_START.md) с примерами запросов к LLM, Eval API и запуска meta-eval пайплайна.

### Meta-Eval Pipeline

Мета-оценка алгоритма улучшения: `metaeval(Alg, D, E, M)`

**Важно:** Перед запуском убедитесь, что Eval API сервер запущен и доступен по адресу `--eval-api-url`.

**Пример 1: С OpenAI-совместимым сервером**

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"

python -m src.pipelines.meta_eval \
    --eval-api-url https://hackathon.intellemma.ru \
    --eval-api-key <ВАШ_EVAL_API_KEY_ИЗ_ПИСЬМА> \
    --dataset-id mtbench101 \
    --num-examples 10 \
    --split val \
    --seed 42 \
    --judge-id golden_semantic_match_v1 \
    --agent-prompt "Ты профессиональный AI-ассистент службы поддержки. Помогай пользователям решать вопросы вежливо и эффективно." \
    --model-name openai/gpt-oss-20b \
    --model-base-url https://provider.intellemma.ru/v1 \
    --model-api-key "$OPENAI_API_KEY" \
    --model-temperature 0.7 \
    --model-max-tokens 1000 \
    --judge-name openai/gpt-oss-20b \
    --judge-base-url https://provider.intellemma.ru/v1 \
    --judge-api-key "$OPENAI_API_KEY" \
    --judge-temperature 0.0 \
    --judge-max-tokens 1500
```

**Пример 2: С OpenAI API**

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"

python -m src.pipelines.meta_eval \
    --eval-api-url https://hackathon.intellemma.ru \
    --eval-api-key <ВАШ_EVAL_API_KEY_ИЗ_ПИСЬМА> \
    --dataset-id mtbench101 \
    --num-examples 10 \
    --split val \
    --seed 42 \
    --judge-id golden_semantic_match_v1 \
    --agent-prompt "Ты профессиональный AI-ассистент службы поддержки..." \
    --model-name gpt-4o-mini \
    --model-temperature 0.7 \
    --model-max-tokens 1000 \
    --judge-name gpt-4o-mini \
    --judge-temperature 0.0 \
    --judge-max-tokens 1500
```

**Что делает:**
1. Подготавливает датасет и считает размер (data_size) через tiktoken
2. Запускает алгоритм улучшения: `Prompt = Alg(D, E, M)`
   - Профилирует токены алгоритма 
   - Вычисляет коэффициенты расхода токенов:
     - `usage_coefficient = alg_total_tokens / data_size` (общий коэффициент)
     - `input_coefficient = alg_prompt_tokens / data_size`
     - `output_coefficient = alg_completion_tokens / data_size`
3. Оценивает полученный промпт: `evaluate(E, D, M, Prompt)` через API
   - Регенерирует ответы на всех `num_examples` диалогах с новым промптом
   - Оценивает их через judge
   - Получает результаты через API и сохраняет их в run директорию meta_eval
4. Сохраняет улучшенный промпт, результаты оценки и статистику токенов

**Результаты:**
- `runs/<run_id>/prompts/agent_prompt_improved.txt` - улучшенный промпт
- `runs/<run_id>/regen/eval_regen.json` - регенерированные диалоги
- `runs/<run_id>/judge/eval_{split}__{judge_id}.json` - оценки диалогов
- `runs/<run_id>/judge/summary_eval.json` - агрегированные скоры
- `runs/<run_id>/meta_eval_results.json` - метаданные и результаты оценки
- `runs/<run_id>/usage/usage_log.jsonl` - лог всех LLM вызовов
- `runs/<run_id>/usage/usage_report.json` - агрегированный отчет по токенам (включая `alg_profiling`)
- `runs/<run_id>/final_report.json` - финальный отчет с:
  - `usage` - токены алгоритма (только improver)
  - `eval_usage` - токены evaluate (regen + judge для всех диалогов)
  - `usage_coefficients` - коэффициенты расхода токенов алгоритма


**Примечание:** Алгоритм `Alg` выбирается в `selector.py` (по умолчанию `BaselineAlg` из `improve_algorithm.py`). Для использования своего алгоритма реализуйте `MyAlg` в `submission.py` и замените `BaselineAlg` на `MyAlg` в `selector.py`.

## ⚙️ Параметры Meta-Eval Pipeline

Параметры соответствуют `metaeval(Alg, D, E, M)`:
- **D (dataset)**: `--dataset-id`, `--num-examples`, `--split`, `--seed`
- **E (evaluator/judge)**: `--eval-api-url`, `--eval-api-key`, `--judge-id`, `--judge-name`, `--judge-temperature`, `--judge-max-tokens`, `--judge-base-url`, `--judge-api-key`
- **M (model)**: `--model-name`, `--model-temperature`, `--model-max-tokens`, `--model-base-url`, `--model-api-key`
- **Alg**: Выбирается в `selector.py` (по умолчанию `BaselineAlg`)

**Обязательные параметры:**
- `--eval-api-url`: URL API для оценки (например, https://hackathon.intellemma.ru)
- `--dataset-id`: ID датасета
- `--num-examples`: Количество примеров
- `--split`: Split датасета (train/val/test)
- `--judge-id`: ID judge промпта
- `--agent-prompt`: Начальный промпт агента
- `--model-name`: Имя модели для генерации ответов
- `--judge-name`: Имя модели для оценки

**Опциональные параметры:**
- `--eval-api-key`: API ключ для авторизации в Eval API (Bearer токен из письма)
- `--seed`: Seed для воспроизводимости (по умолчанию: 42)
- Параметры моделей (temperature, max_tokens, base_url, api_key)

### Реестр датасетов (`data/datasets_registry.json`)

Доступные датасеты:
- `mtbench101` - MT-Bench-101 (train=1248, val=69, test=70)
- `empathetic_dialogues` - EmpatheticDialogues (train=1486, val=498, test=98)
- `multiwoz_2_1` - MultiWOZ 2.1 (train=1476, val=500, test=100)

## 📊 Формат данных

### Датасет (`data/dataset.json`)

```json
{
  "train": {
    "dlg_0001": {
      "turns": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ]
    }
  },
  "val": {...},
  "test": {...}
}
```

### Judge результат

```json
{
  "dlg_0001": {
    "detailed_analysis": "Полный разбор по всем критериям...",
    "detailed_evaluations": {
      "criterion1": "Детальная оценка...",
      "criterion2": "Детальная оценка..."
    },
    "subscores": {
      "criterion1": 8.5,
      "criterion2": 9.0
    },
    "explanation": "Краткое объяснение...",
    "overall": 87.5
  }
}
```

**Примечание:** 
- `subscores` хранятся в шкале 1-10 (исходные значения от judge)
- `overall` вычисляется автоматически как среднее всех `subscores`, умноженное на 10 (100-балльная шкала)
- При агрегации все `subscores` умножаются на 10 для получения 100-балльной шкалы в финальных отчетах

## 📁 Результаты

Результаты сохраняются в `runs/<run_id>/`:

- **`config.json`** - Конфигурация запуска
- **`prompts/`**
  - `agent_prompt_improved.txt` - Улучшенный промпт (для meta_eval)
- **`regen/`**
  - `eval_regen.json` - Регенерированные диалоги
- **`judge/`**
  - `eval_{split}__{judge_id}.json` - Оценки диалогов
  - `summary_eval.json` - Агрегированные скоры
- **`usage/`**
  - `usage_log.jsonl` - Лог всех LLM вызовов
  - `usage_report.json` - Агрегированный отчет по токенам (включая `alg_profiling` с коэффициентами)
- **`final_report.json`** - Финальный отчет с:
  - Скорами оценки (`evaluation`)
  - Токенами алгоритма (`usage`)
  - Токенами evaluate (`eval_usage`)
  - Коэффициентами расхода токенов (`usage_coefficients`):
    - `usage_coefficient` - общий коэффициент (total_tokens / data_size)
    - `input_coefficient` - коэффициент входных токенов (prompt_tokens / data_size)
    - `output_coefficient` - коэффициент выходных токенов (completion_tokens / data_size)

## 🔍 Judge промпты

Система поддерживает 5 judge промптов для оценки по разным критериям:

- **`golden_semantic_match_v1`**: reference-based (с `{golden_answer}`) — семантическое соответствие эталону
- **`instruction_adherence_v1`**: instruction-following — требования/ограничения/формат из `{golden_history}`
- **`helpfulness_problem_solving_v1`**: problem-solving — практическая полезность и продвижение к решению
- **`completeness_coverage_v1`**: completeness — полнота и достаточность деталей
- **`clarity_structure_v1`**: clarity — ясность/структура/лаконичность

⚠️ Важно про тестирование алгоритмов улучшения:
В реальном тестировании и мета-оценке алгоритмов улучшения будут использоваться другие judge промпты, отличающиеся формулировками критериев, шкалой и стилем разборов. Поэтому не стоит “подгонять” алгоритм улучшения под конкретные judge промпты из этого списка (например, оптимизировать текст промпта только ради роста этих конкретных метрик). Алгоритм должен улучшать промпт универсально, чтобы улучшения сохранялись при смене оценщика. 

## 🎓 Как работает оценка (Judge)

Оценка выполняется через внешний Eval API сервер. Judge промпты возвращают:

1. **Полный разбор (detailed_analysis)**: Анализ диалога по всем критериям (2-3 предложения на критерий)
2. **Детальные оценки (detailed_evaluations)**: Детальная оценка по каждому критерию (что конкретно хорошо/плохо)
3. **Числовые оценки (subscores)**: Оценка от 1 до 10 по каждому критерию (внутри judge)
4. **Общая оценка (overall)**: Автоматически вычисляется как среднее всех subscores, умноженное на 10 (100-балльная шкала)
5. **Агрегированные оценки**: При агрегации все subscores умножаются на 10 для получения 100-балльной шкалы

## 🔧 Особенности реализации

- **OpenAI-совместимые API**: Основной режим работы - поддержка локальных серверов (vLLM) через `base_url` и `api_key`. Также поддерживается обычный OpenAI API через `OPENAI_API_KEY`
- **Случайная выборка**: Диалоги выбираются случайно с фиксированным seed для воспроизводимости
- **Внешний Eval API**: Оценка промптов выполняется через внешний API сервер (предоставляется организаторами)
- **Retry логика**: 3 попытки при ошибках LLM, парсинга JSON, пустом ответе или таймауте
- **Таймаут**: 30 секунд на каждый LLM запрос с автоматическим ретраем при таймауте
- **Логирование usage**: Все LLM вызовы логируются с токенами и латентностью
- **Реестр датасетов**: Централизованное управление датасетами через JSON реестр
- **Модульная архитектура алгоритмов**: Легкое переключение между алгоритмами через `selector.py`, возможность создания собственных алгоритмов в `submission.py`

## 📝 API пайплайнов

### Алгоритмы улучшения промпта

Система поддерживает несколько алгоритмов улучшения промпта:

1. **`BaselineAlg`** (в `improve_algorithm.py`) - базовый алгоритм, который:
   - Выбирает 1 случайный диалог из датасета
   - Улучшает промпт на основе этого диалога через LLM
   - Промпт для улучшения захардкожен внутри алгоритма
   - Параметры модели для improver берутся из `model_cfg`

2. **`MyAlg`** (в `submission.py`) - шаблон для пользовательских алгоритмов

3. **`Alg`** (в `selector.py`) - функция выбора алгоритма. По умолчанию использует `BaselineAlg`, но можно переключить на `MyAlg` или любой другой алгоритм.

**Пример использования BaselineAlg:**

```python
from src.pipelines.improve_algorithm import BaselineAlg
from src.core.llm import LLMClient

llm_client = LLMClient(usage_logger=...)

new_prompt, metadata = await BaselineAlg(
    dataset_id="mtbench101",
    num_examples=10,
    split="val",
    judge_cfg={...},  # Не используется в BaselineAlg, но требуется по сигнатуре
    model_cfg={
        "name": "openai/gpt-oss-20b",
        "temperature": 0.5,  # Используется для improver
        "max_tokens": 2000,  # Используется для improver
        "base_url": "https://provider.intellemma.ru/v1",   # Опционально
        "api_key": "..."     # Опционально
    },
    agent_prompt_current="Ты профессиональный AI-ассистент...",
    llm_client=llm_client,
    seed=42
)
```

**Пример создания своего алгоритма:**

1. Реализуйте `MyAlg` в `submission.py` с сигнатурой:
```python
async def MyAlg(
    dataset_id: str,
    num_examples: int,
    split: str,
    judge_cfg: dict,
    model_cfg: dict,
    agent_prompt_current: str,
    llm_client,
    seed: int = 42
) -> tuple[str, dict]:
    # Your code here
    return improved_prompt, metadata
```

2. Переключите алгоритм в `selector.py`:
```python
# Замените BaselineAlg на MyAlg
return await MyAlg(...)
```

### `metaeval(Alg, D, E, M)`

```python
from src.pipelines.meta_eval import metaeval
from src.core.usage import TokenProfiler
from pathlib import Path

profiler = TokenProfiler()
run_path = Path("runs/run_20240101_120000")

results = await metaeval(
    Alg=my_algorithm,
    dataset_id="mtbench101",
    num_examples=10,
    split="val",
    judge_cfg={...},
    model_cfg={...},
    eval_api_url="https://hackathon.intellemma.ru",
    eval_api_key="<ВАШ_EVAL_API_KEY>",
    profiler=profiler,
    run_path=run_path,
    seed=42
)
```
