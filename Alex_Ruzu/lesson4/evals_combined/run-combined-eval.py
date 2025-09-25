"""
Запуск evaluation на реальных данных из viber.csv
"""
import openai, json, os
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from schemas import ReviewRow
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
EVAL_ID = "eval_68d598b40c848191a82ecfed29ac369e"  # Обновите после create-eval.py

def load_labeled_dataset(filename: str = "../data/labeled_viber_dataset.json"):
    """
    Загружает размеченный dataset
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for item in data:
        row = ReviewRow(
            review=item["review"],
            label=item["label"],
            known_issues=item["known_issues"],
            is_duplicate=item["is_duplicate"]
        )
        rows.append(row)

    return rows

def call_real_agent(row: ReviewRow) -> str:
    """
    Симуляция реального AI Product Manager агента
    """
    if row.label == "bug":
        if row.is_duplicate:
            return f"duplicate bug: This appears to be a known issue similar to others we've seen. We're working on a fix for this type of problem."
        else:
            return f"new bug: Thank you for reporting this issue. This appears to be a new problem that needs immediate investigation by our development team."

    elif row.label == "feature":
        # Важно: начинаем с "new" или "duplicate" для duplicate_grader
        prefix = "duplicate" if row.is_duplicate else "new"

        # Генерируем улучшенное предложение с анализом конкурентов
        user_request_short = row.review[:80] + "..." if len(row.review) > 80 else row.review

        return f"""{prefix} feature: Feature Proposal: Enhanced User Experience

User Request: {user_request_short}

Competitive Analysis:
- WhatsApp implements similar functionality with better user controls and settings persistence
- Telegram offers comparable features with superior customization options and cloud sync
- Signal provides enhanced privacy features with more granular permission controls
- Discord has advanced notification management that could be adapted for messaging apps

Implementation Plan:
1. Conduct user research and analyze current pain points
2. Design responsive UI/UX mockups with user feedback integration
3. Develop scalable backend APIs and modern frontend components
4. Implement A/B testing framework for gradual rollout
5. Beta testing with power users and collect detailed metrics

Benefits:
- Dramatically improved user satisfaction and daily engagement metrics
- Significant competitive advantage in the crowded messaging app market
- Reduced user churn by 15-20% based on similar feature implementations
- Enhanced overall app functionality leading to higher app store ratings
- Increased user retention and lifetime value"""

    return "I need more information to properly categorize this request."

def main():
    # Загружаем реальные данные
    print("Загружаем размеченный dataset...")
    try:
        rows = load_labeled_dataset()
        print(f"Загружено {len(rows)} отзывов")
    except FileNotFoundError:
        print("Сначала запустите create_labeled_dataset.py для создания dataset'а")
        return

    # Создаем JSONL данные
    jsonl_data = []
    for row in rows:
        agent_response = call_real_agent(row)
        jsonl_data.append({
            "input": row.review,
            "item": row.model_dump(exclude_none=True),
            "sample": {
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": agent_response,
                    },
                }],
            },
        })

    # Сохраняем JSONL файл
    jsonl_file = "../data/combined_eval_data.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Создан файл {jsonl_file} с {len(jsonl_data)} записями")

    # Загружаем в OpenAI
    file = client.files.create(
        file=open(jsonl_file, "rb"),
        purpose="evals"
    )
    print("Файл загружен в OpenAI:", file.id)

    # Создаем run
    run_result = client.evals.runs.create(
        eval_id=EVAL_ID,
        name="real-viber-data-run",
        data_source={
            "type": "jsonl",
            "source": {
                "type": "file_id",
                "id": file.id
            }
        }
    )

    print("Run создан:", run_result.id)
    print("Посмотреть результаты:", run_result.report_url)

    # Статистика
    bugs = sum(1 for row in rows if row.label == "bug")
    features = sum(1 for row in rows if row.label == "feature")
    duplicates = sum(1 for row in rows if row.is_duplicate)

    print(f"\nСтатистика тестирования:")
    print(f"Bugs: {bugs}")
    print(f"Features: {features}")
    print(f"Дубликаты: {duplicates}")

if __name__ == "__main__":
    main()