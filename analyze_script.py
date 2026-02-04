import pandas as pd
import spacy
from deep_translator import GoogleTranslator
from textblob import TextBlob
from collections import defaultdict, Counter
import sys
import time

# === 1. 维度与关键词定义 ===
ASPECT_MAP = {
    'accuracy': ['accuracy', 'accurate', 'reading', 'measure', 'temp', 'degree', 'calibrat', 'error', 'offset', 'precision'],
    'battery': ['battery', 'power', 'charge', 'drain', 'life', 'die', 'dead', 'recharge'],
    'usability': ['easy', 'use', 'simple', 'handy', 'interface', 'menu', 'button', 'screen', 'display', 'ui'],
    'manual': ['manual', 'instruction', 'direction', 'guide', 'paper', 'read', 'help'],
    'build_quality': ['broke', 'break', 'plastic', 'cheap', 'durab', 'rugged', 'quality', 'feel', 'sturdy'],
    'price_value': ['price', 'cost', 'value', 'money', 'worth', 'deal', 'expensive', 'cheap'],
    'image_quality': ['image', 'picture', 'resolution', 'clear', 'blur', 'pixel', 'screen', 'clarity'],
    'connectivity': ['app', 'bluetooth', 'wifi', 'connect', 'phone', 'data', 'download', 'sync']
}

SCENARIO_MAP = {
    'Home/Cooking': ['kitchen', 'cook', 'pizza', 'oven', 'grill', 'meat', 'food', 'fry', 'pan', 'baking', 'milk', 'baby', 'water'],
    'Automotive': ['car', 'engine', 'tire', 'brake', 'rotor', 'exhaust', 'radiator', 'mechanic', 'auto', 'bearing', 'vehicle'],
    'Industrial/HVAC': ['hvac', 'ac', 'vent', 'air', 'pipe', 'leak', 'insulation', 'wall', 'electric', 'circuit', 'breaker', 'panel', 'weld'],
    'Medical/Body': ['body', 'fever', 'head', 'forehead', 'sick', 'human', 'health']
}

PERSONA_MAP = {
    'Professional (HVAC/Mech)': ['tech', 'technician', 'work', 'job', 'client', 'install', 'shop', 'trade', 'daily', 'pro'],
    'Homeowner/DIY': ['home', 'house', 'project', 'diy', 'hobby', 'husband', 'wife', 'dad', 'gift', 'weekend'],
    'Parent': ['kid', 'child', 'son', 'daughter', 'baby', 'mom', 'nursery']
}

try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def translate_text(text):
    if not isinstance(text, str) or len(text) < 2: return ""
    try:
        # 简单判断是否包含非ASCII字符（例如中文、日文）
        if len(text.encode('utf-8')) != len(text):
            time.sleep(0.1) 
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except:
        return text

def analyze_sentiment_sentence(sentence):
    return TextBlob(sentence).sentiment.polarity

def classify_aspect(sentence_text):
    found_aspects = []
    sentence_lower = sentence_text.lower()
    for aspect, keywords in ASPECT_MAP.items():
        for kw in keywords:
            if kw in sentence_lower:
                found_aspects.append(aspect)
                break
    return found_aspects

def classify_scenario_persona(text):
    text_lower = text.lower()
    found_scenarios = []
    found_personas = []
    for cat, kws in SCENARIO_MAP.items():
        if any(kw in text_lower for kw in kws): found_scenarios.append(cat)
    for cat, kws in PERSONA_MAP.items():
        if any(kw in text_lower for kw in kws): found_personas.append(cat)
    return found_scenarios, found_personas

def main():
    INPUT_FILE = 'reviews.csv'
    OUTPUT_FILE = 'analysis_report.txt'
    
    print("1. 读取数据...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"错误：读取文件失败 - {e}")
        sys.exit(1)

    # 必需列检查
    required_cols = ['Content', 'Title']
    if not all(col in df.columns for col in required_cols):
        print(f"错误：CSV缺少必需列 {required_cols}。请检查表头。")
        sys.exit(1)

    df['full_text'] = df['Title'].fillna('') + ". " + df['Content'].fillna('')
    has_rating = 'Rating' in df.columns
    
    # 统计容器
    aspect_stats = defaultdict(lambda: {'pos': 0, 'neg': 0, 'examples_pos': [], 'examples_neg': []})
    scenario_counts = Counter()
    persona_counts = Counter()
    
    total_reviews = len(df)
    print(f"2. 分析 {total_reviews} 条评论...")

    for idx, row in df.iterrows():
        raw_text = str(row['full_text'])
        text = translate_text(raw_text)
        if not text or len(text) < 3: continue

        scens, pers = classify_scenario_persona(text)
        scenario_counts.update(scens)
        persona_counts.update(pers)
        
        doc = nlp(text)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text) < 5: continue
            
            score = analyze_sentiment_sentence(sent_text)
            sentiment_label = 'neu'
            if score > 0.15: sentiment_label = 'pos'
            elif score < -0.1: sentiment_label = 'neg'
            
            if has_rating and pd.notnull(row['Rating']):
                if row['Rating'] <= 2 and score < 0.1: sentiment_label = 'neg'

            if sentiment_label == 'neu': continue

            aspects = classify_aspect(sent_text)
            for aspect in aspects:
                aspect_stats[aspect][sentiment_label] += 1
                current_examples = aspect_stats[aspect][f'examples_{sentiment_label}']
                if len(current_examples) < 3:
                    current_examples.append(sent_text)
                else:
                    current_examples.sort(key=len)
                    if len(sent_text) > 20 and len(sent_text) < len(current_examples[-1]):
                        current_examples[-1] = sent_text

    # === 生成报告 ===
    lines = []
    lines.append(f"=== 亚马逊评论深度分析报告 (N={total_reviews}) ===\n")
    
    lines.append("一、产品核心问题 (Core Problems)")
    sorted_neg = sorted(aspect_stats.items(), key=lambda x: x[1]['neg'], reverse=True)
    for aspect, data in sorted_neg:
        count = data['neg']
        if count == 0: continue
        ratio = (count / total_reviews) * 100
        lines.append(f"🔴 维度: [{aspect.upper()}] - 频次: {count} | 占比: {ratio:.1f}%")
        lines.append(f"   佐证: \"{data['examples_neg'][0] if data['examples_neg'] else '无'}\"")

    lines.append("\n二、主要卖点 (Key Selling Points)")
    sorted_pos = sorted(aspect_stats.items(), key=lambda x: x[1]['pos'], reverse=True)
    for aspect, data in sorted_pos:
        count = data['pos']
        if count == 0: continue
        ratio = (count / total_reviews) * 100
        lines.append(f"🟢 维度: [{aspect.upper()}] - 频次: {count} | 占比: {ratio:.1f}%")
        lines.append(f"   佐证: \"{data['examples_pos'][0] if data['examples_pos'] else '无'}\"")

    lines.append("\n三、高频应用场景 (Top Scenarios)")
    if scenario_counts:
        for scen, count in scenario_counts.most_common(5):
            ratio = (count / total_reviews) * 100
            lines.append(f"   - {scen}: {count}次 (占比 {ratio:.1f}%)")
    
    lines.append("\n四、推断用户画像 (Inferred Personas)")
    if persona_counts:
        for pers, count in persona_counts.most_common(5):
            ratio = (count / total_reviews) * 100
            lines.append(f"   - {pers}: {count}次 (占比 {ratio:.1f}%)")

    report_text = "\n".join(lines)
    print(report_text)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)

if __name__ == "__main__":
    main()
