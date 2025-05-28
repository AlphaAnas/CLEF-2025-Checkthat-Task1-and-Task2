import os
import uuid
import time
import random
import pandas as pd
from google import genai
from dotenv import load_dotenv
import json

# Load API Key
load_dotenv()


api_key = os.getenv("api_key_gemini")
model = genai.Client(api_key=api_key)

# Constants
MIN_DATASET_SIZE = 250  # Minimum dataset size per language
SENTENCES_PER_REQUEST = 50  # Number of sentences to generate in each API call
MAX_RETRIES = 5         # Maximum number of retry attempts
RETRY_DELAY = 4         # Base delay in seconds between retries

# Topics to diversify the dataset
TOPICS = [
    "politics", "technology", "science", "education", 
    "health", "economy", "environment", "culture",
    "sports", "entertainment", "social_media", "travel",
    "food", "fashion", "literature", "history"
]

# Enhance logging
def log(msg, level="INFO"):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")

# Retry-safe generation function with exponential backoff
def generate_content(prompt, retries=MAX_RETRIES, delay=RETRY_DELAY):
    for attempt in range(retries):
        try:
            response = model.models.generate_content(
                            model = "gemini-2.5-flash-preview-04-17",
                            contents=[prompt]
                            )
            return response.text.strip()
        except Exception as e:
            backoff = delay * (2 ** attempt) + random.uniform(0, 1)
            log(f"API Error: {str(e)[:100]}... retrying in {backoff:.1f}s (attempt {attempt+1}/{retries})", "ERROR")
            time.sleep(backoff)
    log(f"Failed after {retries} attempts", "ERROR")
    return ""

# Carefully parse generated content to handle potential formatting issues
def parse_generated_lines(content, expected_label):
    lines = []
    
    # Try different parsing strategies
    if "|" in content:
        # Standard parsing with pipe separator
        raw_lines = [line.strip() for line in content.split("\n") if line.strip()]
        for line in raw_lines:
            if "|" in line:
                try:
                    text, label = line.rsplit("|", 1)
                    text = text.strip()
                    label = label.strip().upper()
                    if label in ["OBJ", "SUBJ"] and len(text) > 10:
                        lines.append((text, label))
                except Exception:
                    # If parsing fails, try to salvage the line with expected label
                    possible_text = line.split("|")[0].strip()
                    if len(possible_text) > 10:
                        lines.append((possible_text, expected_label))
            else:
                # Line without separator but still valid text
                if len(line) > 10:
                    lines.append((line, expected_label))
    else:
        # Fallback: treat each line as a separate sentence
        raw_lines = [line.strip() for line in content.split("\n") if line.strip()]
        lines = [(line, expected_label) for line in raw_lines if len(line) > 10]
    
    return lines

# Dynamic prompt templates with topic focus and larger batch size
def get_prompt(language, label_type, topics=None):
    topic_context = ""
    if topics:
        topic_list = ", ".join(topics)
        topic_context = f" covering the following topics: {topic_list}"
    
    base_prompts = {
        "ar": {
            "OBJ": f"""
اكتب {SENTENCES_PER_REQUEST} جملة موضوعية باللغة العربية{topic_context}. يجب أن تكون مبنية على حقائق قابلة للتحقق وتشبه أسلوب الأخبار أو المقالات الرسمية.

تعليمات مهمة:
1. كل جملة يجب أن تكون مستقلة بذاتها
2. يجب أن تكون الجملة المكتوبة موضوعية تماماً بدون رأي شخصي
3. استخدم جمل متنوعة الطول والتركيب
4. تجنب تكرار نفس الأفكار
5. أضف في نهاية كل جملة الرمز: | OBJ
6. اكتب كل جملة في سطر منفصل

أمثلة:
- "وصل عدد سكان القاهرة إلى أكثر من 20 مليون نسمة في عام 2023. | OBJ"
- "تحتل الصين المرتبة الأولى عالمياً في إنتاج الطاقة الشمسية. | OBJ"
- "أعلنت منظمة الصحة العالمية عن برنامج جديد لمكافحة الأمراض المعدية في أفريقيا. | OBJ"

قم بإنشاء {SENTENCES_PER_REQUEST} جملة موضوعية مختلفة الآن. تذكر أن تنوع في المواضيع والمحتوى لكل جملة:
""",
            "SUBJ": f"""
اكتب {SENTENCES_PER_REQUEST} جملة ذات طابع شخصي أو تعبيري باللغة العربية{topic_context}. يجب أن تعكس آراء أو مواقف شخصية، وتشبه الأسلوب الصحفي التحليلي أو المدونات.

تعليمات مهمة:
1. كل جملة يجب أن تكون مستقلة بذاتها
2. يجب أن تعبر الجملة عن رأي شخصي أو انطباع ذاتي
3. استخدم عبارات مثل "أعتقد" و"يبدو لي" و"من وجهة نظري"
4. تنوع في الآراء والمواقف المعبر عنها
5. أضف في نهاية كل جملة الرمز: | SUBJ
6. اكتب كل جملة في سطر منفصل

أمثلة:
- "أرى أن الاستثمار في التعليم هو أفضل خيار للدول النامية في الوقت الراهن. | SUBJ"
- "يبدو لي أن السياسات الاقتصادية الحالية لن تحقق النتائج المرجوة. | SUBJ"
- "من المؤسف حقاً رؤية هذا التدهور المستمر في مستوى الخدمات الصحية. | SUBJ"

قم بإنشاء {SENTENCES_PER_REQUEST} جملة شخصية مختلفة الآن. تذكر أن تنوع في المواضيع والمحتوى لكل جملة:
"""
        },
        "bg": {
            "OBJ": f"""
Генерирай {SENTENCES_PER_REQUEST} обективни изречения на български език{' за следните теми: ' + topic_list if topics else ''}. Изреченията трябва да представят факти и да наподобяват новинарски стил или енциклопедичен тон.

Важни инструкции:
1. Всяко изречение трябва да бъде независимо
2. Изреченията трябва да съдържат проверими факти
3. Използвай разнообразни теми и структури
4. Избягвай повторения на идеи или формулировки
5. Добави | OBJ в края на всяко изречение
6. Напиши всяко изречение на отделен ред

Примери:
- "Българската писменост е създадена през 9-ти век от братята Кирил и Методий. | OBJ"
- "Европейският съюз въведе нови регулации за дигиталните услуги през 2024 година. | OBJ"
- "Стара планина разделя България на северна и южна част. | OBJ"

Моля, създай {SENTENCES_PER_REQUEST} различни обективни изречения сега, като се стараеш да покриеш разнообразни теми:
""",
            "SUBJ": f"""
Генерирай {SENTENCES_PER_REQUEST} субективни изречения на български език{' за следните теми: ' + topic_list if topics else ''}. Изреченията трябва да отразяват лични мнения, чувства или оценки. Стилът трябва да напомня на публицистика или блог.

Важни инструкции:
1. Всяко изречение трябва да бъде независимо
2. Изреченията трябва да съдържат лични мнения и оценки
3. Използвай изрази като "според мен", "смятам че", "чувствам"
4. Разнообразявай изразените гледни точки
5. Добави | SUBJ в края на всяко изречение
6. Напиши всяко изречение на отделен ред

Примери:
- "Според мен новите технологични разработки променят обществото твърде бързо. | SUBJ"
- "Чувствам се разочарован от липсата на амбиция в културната политика на държавата. | SUBJ"
- "Смятам, че туристическият потенциал на България остава недостатъчно развит. | SUBJ"

Моля, създай {SENTENCES_PER_REQUEST} различни субективни изречения сега, като се стараеш да покриеш разнообразни теми:
"""
        }
    }
    
    return base_prompts[language][label_type]

# Generate a batch of sentences with multi-topic approach
def generate_batch(language, label_type, num_topics=3):
    # Select random topics to focus on
    selected_topics = random.sample(TOPICS, min(num_topics, len(TOPICS)))
    
    # Generate with topic guidance
    prompt = get_prompt(language, label_type, selected_topics)
    content = generate_content(prompt)
    
    if not content:
        log(f"Failed to generate content for {language}/{label_type}", "WARNING")
        return []
        
    lines = parse_generated_lines(content, label_type)
    log(f"Generated {len(lines)} {label_type} sentences for {language} with topics: {', '.join(selected_topics)}")
    return lines

# Main function to create and save dataset
def create_dataset(language_code, min_samples=MIN_DATASET_SIZE, output_path=None):
    if not output_path:
        output_path = f"{language_code}_train_expanded.tsv"
    
    log(f"Starting dataset generation for {language_code.upper()}")
    all_data = []
    
    # Generate data until we reach the minimum dataset size
    while len(all_data) < min_samples:
        # Alternate between objective and subjective sentences
        label_type = "OBJ" if len(all_data) % 2 == 0 else "SUBJ"
        
        # Generate a batch with varying topic focus each time
        new_data = generate_batch(language_code, label_type, num_topics=3)
        all_data.extend(new_data)
        
        log(f"Progress: {len(all_data)}/{min_samples} sentences collected")
    
    # Process and save final dataset
    rows = []
    for sentence, label in all_data:
        sentence_id = str(uuid.uuid4())
        rows.append((sentence_id, sentence, label))
    
    # Shuffle the dataset
    random.shuffle(rows)
    
    # Save as TSV
    df = pd.DataFrame(rows, columns=["sentence_id", "sentence", "label"])
    df.to_csv(output_path, sep="\t", index=False)
    
    # Generate statistics
    stats = {
        "language": language_code,
        "total_samples": len(df),
        "objective_samples": len(df[df["label"] == "OBJ"]),
        "subjective_samples": len(df[df["label"] == "SUBJ"]),
        "avg_sentence_length": df["sentence"].str.len().mean()
    }
    
    log(f"Dataset statistics: {json.dumps(stats, indent=2)}")
    log(f"Successfully saved {len(df)} rows to {output_path}")
    
    return df

# Function to validate dataset quality
def validate_dataset(df, language):
    """Basic validation of dataset quality"""
    log(f"Validating {language} dataset with {len(df)} samples")
    
    # Check label distribution
    label_counts = df["label"].value_counts().to_dict()
    log(f"Label distribution: {label_counts}")
    
    # Check for balanced dataset
    label_ratio = min(label_counts.values()) / max(label_counts.values())
    if label_ratio < 0.7:
        log(f"Warning: Dataset is imbalanced (ratio: {label_ratio:.2f})", "WARNING")
    
    # Check for duplicates
    duplicate_count = len(df) - len(df["sentence"].unique())
    if duplicate_count > 0:
        log(f"Warning: Found {duplicate_count} duplicate sentences", "WARNING")
    
    # Check sentence length
    short_sentences = df[df["sentence"].str.len() < 20]
    if len(short_sentences) > 0:
        log(f"Warning: Found {len(short_sentences)} very short sentences", "WARNING")
    
    return True

# Enhanced data generation with multiple topic batches
def create_enhanced_dataset(language_code, min_samples=MIN_DATASET_SIZE, output_path=None):
    """Create a more diverse dataset by using multiple topic-focused batches"""
    if not output_path:
        output_path = f"{language_code}_train_expanded.tsv"
    
    log(f"Starting enhanced dataset generation for {language_code.upper()}")
    all_data = []
    
    # First generate general content
    for label_type in ["OBJ", "SUBJ"]:
        prompt = get_prompt(language_code, label_type)
        content = generate_content(prompt)
        lines = parse_generated_lines(content, label_type)
        all_data.extend(lines)
        log(f"Generated {len(lines)} general {label_type} sentences")
    
    # Then generate topic-specific content
    topic_groups = [TOPICS[i:i+4] for i in range(0, len(TOPICS), 4)]
    for topic_group in topic_groups:
        if len(all_data) >= min_samples:
            break
            
        for label_type in ["OBJ", "SUBJ"]:
            prompt = get_prompt(language_code, label_type, topic_group)
            content = generate_content(prompt)
            lines = parse_generated_lines(content, label_type)
            all_data.extend(lines)
            log(f"Generated {len(lines)} {label_type} sentences for topics: {', '.join(topic_group)}")
            
            if len(all_data) >= min_samples:
                break
    
    # Process and save final dataset
    rows = []
    for sentence, label in all_data:
        sentence_id = str(uuid.uuid4())
        rows.append((sentence_id, sentence, label))
    
    # Shuffle the dataset
    random.shuffle(rows)
    
    # Save as TSV
    df = pd.DataFrame(rows, columns=["sentence_id", "sentence", "label"])
    df.to_csv(output_path, sep="\t", index=False)
    
    # Generate statistics
    stats = {
        "language": language_code,
        "total_samples": len(df),
        "objective_samples": len(df[df["label"] == "OBJ"]),
        "subjective_samples": len(df[df["label"] == "SUBJ"]),
        "avg_sentence_length": df["sentence"].str.len().mean()
    }
    
    log(f"Dataset statistics: {json.dumps(stats, indent=2)}")
    log(f"Successfully saved {len(df)} rows to {output_path}")
    
    return df

# Main execution
if __name__ == "__main__":
    log(f"Starting data generation process")
    
    # Generate Arabic dataset using enhanced method
    arabic_df = create_enhanced_dataset("ar", MIN_DATASET_SIZE, "arabic_train_expanded1.tsv")
    validate_dataset(arabic_df, "Arabic")
    
    # Generate Bulgarian dataset using enhanced method
    bulgarian_df = create_enhanced_dataset("bg", MIN_DATASET_SIZE, "bulgarian_train_expanded1.tsv")
    validate_dataset(bulgarian_df, "Bulgarian")
    
    log("All datasets have been generated successfully")