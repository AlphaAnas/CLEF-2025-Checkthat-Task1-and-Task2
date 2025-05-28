import os
import time
import random
import pandas as pd
from google import genai  # Gemini package
from openpyxl import Workbook
from openpyxl.styles import Alignment
import shutil
from dotenv import load_dotenv, dotenv_values 

# Loading variables from .env file
load_dotenv() 

# Configure Gemini API
api_key = os.getenv("api_key_gemini")
client = genai.Client(api_key=api_key)

# Prompt templates for generating data
prompts = [
    # Basic dataset prompts
    """
    Generate 50 objective statements that present verifiable facts, statistics, or direct quotations. Follow these examples:
    
    - "The population of China is over one billion people"
    - "The average distance from the Earth to the Moon is about 384,400 kilometers."
    - "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions."
    - "The standard measurement unit for force in the International System of Units is the Newton."
    
    Make sure each statement could be independently verified and avoids expressing opinions or judgments.
    Label each statement as 'OBJ' at the end.
    Format: Statement | OBJ
    """,
    
    """
    Generate 50 subjective statements that express opinions, judgments, or evaluations. Follow these examples:
    
    - "Large corporations are inherently driven by profit motives that often override ethical considerations."
    - "Small businesses are the true engine of the economy and are criminally undervalued."
    - "That popular movie franchise has devolved into nothing more than a cynical cash grab."
    - "The internet, paradoxically, seems to have made many people less informed and more gullible"
    
    Ensure each statement conveys a personal viewpoint, evaluation, or judgment rather than a verifiable fact.
    Label each statement as 'SUBJ' at the end.
    Format: Statement | SUBJ
    """,
    
    # Topic-specific prompts
    """
    Generate 25 objective statements about government, economics, and public policy that present neutral facts without expressing opinions. Follow these patterns:
    
    - "In states with shortages, it's also far more difficult to find teachers for math, science, and special education classes."
    - "To plug budgetary holes, 80 out of 111 cities tracked by Southern Weekly, a mainland newspaper, increased the amount they collected in fines last year."
    - "The fiscal swing was more like 4% of gdp in the two years from 2008 to 2010."
    
    Ensure statements are factual and avoid value judgments.
    Label each statement as 'OBJ' at the end.
    Format: Statement | OBJ
    """,
    
    """
    Generate 25 subjective statements about government, economics, and public policy that express clear opinions or judgments. Follow these patterns:
    
    - "A government serious about equality and 'levelling up' would be looking to do the opposite of what this Bill does."
    - "They have not only tolerated but given encouragement to an ever-expanding cost of government."
    - "If it consolidates power into the hands of the government, we can expect the situation to be pushed hard."
    
    Make sure statements contain clear evaluative language or political perspectives.
    Label each statement as 'SUBJ' at the end.
    Format: Statement | SUBJ
    """,
    
    # Style-mirroring prompts
    """
    Generate 25 objective statements using specific journalistic formats like:
    - Statistical claims (with percentages and numbers)
    - Attribution patterns ("X said Y")
    - Descriptions of events or situations
    
    Examples:
    - "With a big police presence, Pride finally went ahead peacefully in 2014."
    - "China's gdp in 2023 could be more than $2trn below the level forecast in January, reckons Goldman Sachs, another bank."
    - "Marko Mihailović, the 29-year-old figurehead of Belgrade Pride, led the city's winning bid."
    
    Label each statement as 'OBJ' at the end.
    Format: Statement | OBJ
    """,
    
    """
    Generate 25 subjective statements that use rhetorical techniques common in opinion writing:
    - Rhetorical questions
    - Value-laden adjectives
    - Appeals to values or principles
    - First-person collective references ("we")
    
    Examples:
    - "Who so mean that he will not himself be taxed, who so mindful of wealth that he will not favor increasing the popular taxes, in aid of these defective children?"
    - "This is the strongest case for stakeholder capitalism."
    - "Suppose they did work, the tide rising to save and redeem them, and that we should be able to perform the terrific gymnastic feat of getting back our equilibrium."
    
    Label each statement as 'SUBJ' at the end.
    Format: Statement | SUBJ
    """,
    
    # Challenging edge cases
    """
    Generate 15 statements that initially appear objective but contain subtle subjective elements. These should be difficult to classify:
    
    Examples:
    - "Many journalists are trying to cling to the remnants of their professional standards."
    - "The event, which organisers had envisaged as a celebration of a new, progressive era, turned into a chaotic nightmare."
    - "Usually, when politicians use these open, Newspeak-type terms, it is because they are trying to obscure more than they want to say."
    
    These statements should blend factual presentation with subtle opinion markers.
    Label each statement as 'SUBJ' at the end.
    Format: Statement | SUBJ
    """,
    
    """
    Generate 5 statements that appear to express opinions but are actually reporting others' viewpoints or are citing facts that might be mistaken for opinions:
    
    Examples:
    - "Thoughtful proponents of stakeholder capitalism argue that Friedman missed an important point: corporations do not exist in the state of nature, but exist only because society permits them to do so."
    - "Conservatives reject that.' Under Boris Johnson, the term has been more clearly used in the positive sense, dropping the reference to levelling down."
    
    Label each statement as 'OBJ' at the end.
    Format: Statement | OBJ
    """
]

# Domain-specific prompt
domain_prompt = """
Generate 30 statements equally distributed across these domains:
1. Politics and governance
2. Healthcare and public health
3. Economics and finance
4. Social issues and equality
5. Legal matters

Make half of them (15) objective statements labeled 'OBJ' and half (15) subjective statements labeled 'SUBJ'.
Objective statements should present verifiable facts without opinions.
Subjective statements should express clear opinions or judgments.

Format each line as: Statement | LABEL (where LABEL is either OBJ or SUBJ)
"""

def generate_with_gemini(prompt, model="gemini-2.5-flash-preview-04-17"):
    """Generate text using Gemini API with error handling and rate limiting"""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            
            response = client.models.generate_content(
                model = "gemini-2.5-flash-preview-04-17",
                contents=[prompt]
                )

            
            if hasattr(response, 'text'):
                return response.text
            else:
                print(f"No text attribute in response: {response}")
                return ""
                
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed with error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("All retry attempts failed.")
                return ""
    
    return ""

def parse_generated_text(text):
    """Parse the generated text into a list of (statement, label) tuples"""
    results = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('Format:') or line.startswith('-'):
            continue
            
        # Check if the line contains a separator and label
        if '|' in line:
            parts = line.split('|')
            if len(parts) >= 2:
                statement = parts[0].strip()
                label = parts[1].strip()
                # Clean up any extra formatting that might be present
                if label.upper() in ['OBJ', 'SUBJ']:
                    results.append((statement, label.upper()))
        
        # Handle cases where Gemini outputs numbered lists
        elif '. ' in line and len(line.split('. ', 1)) == 2:
            # Try to extract the label from the end
            content = line.split('. ', 1)[1]
            if ' OBJ' in content:
                statement = content.replace(' OBJ', '')
                results.append((statement, 'OBJ'))
            elif ' SUBJ' in content:
                statement = content.replace(' SUBJ', '')
                results.append((statement, 'SUBJ'))
    
    return results

def generate_and_save_data(output_directory):
    """Generate data using all prompts and save to CSV and TSV"""
    all_data = []
    
    # Process each prompt
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}...")
        generated_text = generate_with_gemini(prompt)
        parsed_data = parse_generated_text(generated_text)
        all_data.extend(parsed_data)
        
        # Prevent rate limiting
        if i < len(prompts) - 1:
            time.sleep(2)
    
    # Process domain-specific prompt
    print("Processing domain-specific prompt...")
    domain_text = generate_with_gemini(domain_prompt)
    domain_data = parse_generated_text(domain_text)
    all_data.extend(domain_data)
    
    # Create a DataFrame
    df = pd.DataFrame(all_data, columns=['sentence', 'label'])
    
    # Add sentence_id column (UUID-like)
    df['sentence_id'] = [f"gen-{random.randint(100000, 999999)}-{i}" for i in range(len(df))]
    
    # Add solved_conflict column (all False since these are generated)
    df['solved_conflict'] = False
    
    # Reorder columns to match original data
    df = df[['sentence_id', 'sentence', 'label', 'solved_conflict']]
    
    # Save to CSV
    csv_output_file = os.path.join(output_directory, "gemini_generated_data4.csv")
    df.to_csv(csv_output_file, index=False)
    print(f"Generated {len(df)} examples and saved to {csv_output_file}")
    
    # Save to TSV
    tsv_output_file = os.path.join(output_directory, "gemini_generated_data4.tsv")
    df.to_csv(tsv_output_file, sep='\t', index=False)
    print(f"Also saved the data to {tsv_output_file}")
    
    return df

def main():
    # Set directories
    output_directory = r"./data/augmented"
    os.makedirs(output_directory, exist_ok=True)
    
    print("Starting data generation with Gemini...")
    generated_data = generate_and_save_data(output_directory)
    
    # Print statistics
    obj_count = sum(generated_data['label'] == 'OBJ')
    subj_count = sum(generated_data['label'] == 'SUBJ')
    print(f"Generated data statistics:")
    print(f"Total examples: {len(generated_data)}")
    print(f"Objective examples: {obj_count} ({obj_count/len(generated_data)*100:.1f}%)")
    print(f"Subjective examples: {subj_count} ({subj_count/len(generated_data)*100:.1f}%)")
    
    # Create combined version with original data
    try:
        # Check if there's original data to combine with
        original_data_path = "./data/english/data.tsv"  # Adjust path as needed
        if os.path.exists(original_data_path):
            print(f"Found original data at {original_data_path}. Creating combined dataset...")
            original_df = pd.read_csv(original_data_path, sep='\t')
            combined_df = pd.concat([original_df, generated_data], ignore_index=True)
            
            # Save combined dataset
            # combined_csv_path = os.path.join(output_directory, "combined_data.csv")
            combined_tsv_path = os.path.join(output_directory, "combined_data.tsv")
            
            # combined_df.to_csv(combined_csv_path, index=False)
            combined_df.to_csv(combined_tsv_path, sep='\t', index=False)
            
            print(f"Combined dataset created with {len(combined_df)} examples")
            print(f"- Original examples: {len(original_df)}")
            print(f"- Generated examples: {len(generated_data)}")
    except Exception as e:
        print(f"Could not create combined dataset: {e}")

if __name__ == "__main__":
    main()