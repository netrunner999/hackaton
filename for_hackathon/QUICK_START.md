## ⚡ Quick Start


### 0. Запуск в 1 команду (токены уже захардкожены в `meta_eval.py`)

После установки зависимостей можно запустить пайплайн вообще без аргументов:

```bash
python -m src.pipelines.meta_eval
```

Если хотите переопределить только часть параметров, передайте нужные флаги (например, `--dataset-id`, `--num-examples`).

### 1. Быстрый запрос к LLM (через OpenAI-совместимый API)

Простейший пример запроса к LLM через OpenAI-совместимый сервер Intellemма:

```bash
curl --location 'https://provider.intellemma.ru/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer 0726dd7e0fff495120de20e640be4cd341bb7720fdb9d77101e88975e2870e82' \
  --data '{
    "model": "openai/gpt-oss-20b",
    "messages": [
      { "role": "system", "content": "You are helpful assistant. REASONING : LOW" },
      { "role": "user", "content": "Hello! i need you to show me how llm works, in one sentence" }
    ]
  }'
```

### 2. Быстрый старт с Eval API

Для работы LLM-клиента укажите OpenAI-совместимый ключ:

```bash
export OPENAI_API_KEY="0726dd7e0fff495120de20e640be4cd341bb7720fdb9d77101e88975e2870e82"
```

- **Ссылка на Eval API**: `https://hackathon.intellemma.ru`
- **Документация**: `https://hackathon.intellemma.ru/docs`

Запрос к Eval API выполняется с заголовком:

```http
Authorization: Bearer <ВАШ_EVAL_API_KEY_ИЗ_ПИСЬМА>
```

Пример тела запроса:

```json
{
  "dataset_id": "mtbench101",
  "num_examples": 1,
  "split": "train",
  "seed": 42,
  "judge_id": "golden_semantic_match_v1",
  "agent_prompt": "string",
  "model": {
    "name": "openai/gpt-oss-20b",
    "temperature": 0.7,
    "max_tokens": 1000,
    "base_url": "https://provider.intellemma.ru/v1"
  },
  "judge": {
    "name": "openai/gpt-oss-20b",
    "temperature": 0.7,
    "max_tokens": 1000,
    "base_url": "https://provider.intellemma.ru/v1"
  }
}
```

### 3. Быстрый старт с Meta Eval Pipeline

Запустите meta-eval пайплайн:

```bash
python -m src.pipelines.meta_eval \
    --eval-api-url https://hackathon.intellemma.ru \
    --eval-api-key <ВАШ_EVAL_API_KEY_ИЗ_ПИСЬМА> \
    --dataset-id mtbench101 \
    --num-examples 10 \
    --split val \
    --seed 42 \
    --judge-id golden_semantic_match_v1 \
    --agent-prompt "Ты профессиональный AI-ассистент службы поддержки." \
    --model-name openai/gpt-oss-20b \
    --model-base-url https://provider.intellemma.ru/v1 \
    --model-temperature 0.7 \
    --model-max-tokens 1000 \
    --judge-name openai/gpt-oss-20b \
    --judge-base-url https://provider.intellemma.ru/v1 \
    --judge-temperature 0.0 \
    --judge-max-tokens 1500
```

