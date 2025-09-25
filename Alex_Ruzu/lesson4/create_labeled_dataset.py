"""
Скрипт для создания размеченного dataset'а из viber.csv
"""
import pandas as pd
import json
import sys
import os
# No need to modify sys.path since we're in the root directory
from schemas import ReviewRow
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def classify_review_with_ai(content: str) -> dict:
    """
    Использует GPT для автоматической классификации отзыва
    """
    prompt = f"""
    Классифицируй этот отзыв пользователя о приложении Viber:

    "{content}"

    Определи:
    1. Тип: "bug" (если это сообщение о проблеме/ошибке) или "feature" (если это запрос новой функции)
    2. Дубликат: true/false (является ли это распространенной проблемой)

    Ответь в JSON формате:
    {{
        "label": "bug" или "feature",
        "is_duplicate": true или false,
        "reasoning": "краткое объяснение"
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        result = json.loads(response.choices[0].message.content.strip())
        return result
    except:
        # Fallback classification
        content_lower = content.lower()
        is_bug = any(word in content_lower for word in [
            'error', 'crash', 'bug', 'problem', 'issue', 'broken', 'fail',
            'glitch', 'freeze', 'not work', 'stuck'
        ])

        return {
            "label": "bug" if is_bug else "feature",
            "is_duplicate": False,
            "reasoning": "Automatic fallback classification"
        }

def create_evaluation_dataset(csv_file: str, sample_size: int = 20):
    """
    Создает размеченный dataset для evaluation
    """
    # Читаем CSV
    df = pd.read_csv(csv_file)

    # Берем выборку
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)

    labeled_data = []

    print(f"Обрабатываем {len(sample_df)} отзывов...")

    for idx, row in sample_df.iterrows():
        content = row['content']
        print(f"\nОбрабатываем: {content[:50]}...")

        # Автоматическая классификация
        classification = classify_review_with_ai(content)

        # Создаем ReviewRow
        review_row = ReviewRow(
            review=content,
            label=classification["label"],
            known_issues=[] if classification["label"] == "feature" else ["Similar issue reported"],
            is_duplicate=classification["is_duplicate"]
        )

        labeled_data.append({
            "review_row": review_row,
            "original_score": row['score'],
            "classification": classification
        })

        print(f"  → {classification['label']}, duplicate: {classification['is_duplicate']}")
        print(f"  → {classification['reasoning']}")

    return labeled_data

def save_labeled_dataset(labeled_data, filename: str = "labeled_viber_dataset.json"):
    """
    Сохраняет размеченный dataset
    """
    export_data = []
    for item in labeled_data:
        export_data.append({
            "review": item["review_row"].review,
            "label": item["review_row"].label,
            "known_issues": item["review_row"].known_issues,
            "is_duplicate": item["review_row"].is_duplicate,
            "original_score": item["original_score"],
            "reasoning": item["classification"]["reasoning"]
        })

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"\nДанные сохранены в {filename}")
    return filename

if __name__ == "__main__":
    # Создаем размеченный dataset
    csv_path = "data/viber.csv"
    labeled_data = create_evaluation_dataset(csv_path, sample_size=15)

    # Сохраняем
    dataset_file = save_labeled_dataset(labeled_data, "data/labeled_viber_dataset.json")

    # Статистика
    bugs = sum(1 for item in labeled_data if item["review_row"].label == "bug")
    features = sum(1 for item in labeled_data if item["review_row"].label == "feature")
    duplicates = sum(1 for item in labeled_data if item["review_row"].is_duplicate)

    print(f"\nСтатистика dataset'а:")
    print(f"Всего отзывов: {len(labeled_data)}")
    print(f"Bugs: {bugs}")
    print(f"Features: {features}")
    print(f"Дубликаты: {duplicates}")