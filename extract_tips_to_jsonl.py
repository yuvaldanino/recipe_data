import json
import re
from recipe_scrapers import scrape_me

sample_urls = [
    # Previously working URLs
    "https://www.allrecipes.com/recipe/24074/alysias-basic-meat-lasagna/",
    "https://www.bbcgoodfood.com/recipes/best-spaghetti-bolognese-recipe",
    "https://www.simplyrecipes.com/recipes/perfect_guacamole/",
    # New URLs provided by user
    "https://www.allrecipes.com/recipe/158968/spinach-and-feta-turkey-burgers/",
    "https://www.bbcgoodfood.com/recipes/chicken-tikka-masala",
    "https://www.bonappetit.com/recipe/perfect-steak",
    "https://www.seriouseats.com/perfect-scrambled-eggs-recipe",
    "https://www.foodnetwork.com/recipes/ina-garten/roast-chicken-recipe-1940592",
    "https://www.epicurious.com/recipes/food/views/classic-lasagna-51249010",
    "https://cookieandkate.com/best-vegan-chili-recipe/",
    "https://www.ambitiouskitchen.com/healthy-banana-bread/",
    "https://www.feastingathome.com/vegetarian-enchiladas/",
    "https://www.downshiftology.com/recipes/creamy-tuscan-chicken/"
]

# Expanded prompt templates
prompt_templates = [
    "What's a tip for making {title}?",
    "How do I make {title} better?",
    "Any advice for cooking {title}?",
    "How do I avoid common mistakes with {title}?",
    "What should I know before making {title}?",
    "How do I get the best results with {title}?",
    "What's important to remember when making {title}?"
]
# General prompts (not recipe-title-based)
general_prompt_templates = [
    "What's a tip for cooking {main_ingredient}?",
    "How do I avoid dry {main_ingredient}?",
    "How do I get the best results when cooking {main_ingredient}?"
]

# Expanded actionable keywords
actionable_keywords = [
    "always", "never", "make sure", "let", "avoid", "important", "tip", "ensure", "best", "don't", "do not", "should", "must", "recommend", "careful", "keep", "use", "try", "allow", "rest", "cook", "bake", "stir", "whisk", "add", "remove", "check", "until", "before", "after", "because", "so that", "in order to", "to prevent", "to avoid", "to ensure", "why", "how"
]

def is_actionable(sentence):
    s = sentence.lower()
    # Must contain a keyword and be at least 6 words
    if not (any(kw in s for kw in actionable_keywords) and len(s.split()) > 5):
        return False
    # Prefer sentences that explain why/how
    if any(x in s for x in ["because", "so that", "in order to", "to prevent", "to avoid", "to ensure", "why", "how"]):
        return True
    # Or imperative sentences (start with a verb)
    if re.match(r'^(let|add|use|keep|make|avoid|cook|bake|stir|whisk|check|allow|rest|remove|try|ensure|spray|transfer|place|prepare|microwave|blend|whisk|scatter|heat|cover|uncover|drain|increase|reduce|put|saute|assemble|preheat|sprinkle|garnish|remember|start|spray|transfer|mix|form|cook|insert|scatter|remove|tip|recommend|must|should|never|always|careful|important|best|don\'t|do not)\b', s):
        return True
    return False

def extract_sentences(text):
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def extract_main_ingredient(title, tips):
    # Try to guess the main ingredient from the title or tips
    # Simple heuristic: look for common food words in title or first tip
    food_words = ["chicken", "beef", "pasta", "egg", "eggs", "bread", "banana", "lasagna", "turkey", "spinach", "feta", "guacamole", "chili", "lentil", "soup", "enchilada", "steak", "rice", "cheese", "vegetable", "vegan"]
    title_lower = title.lower()
    for word in food_words:
        if word in title_lower:
            return word
    for tip in tips:
        tip_lower = tip.lower()
        for word in food_words:
            if word in tip_lower:
                return word
    return "the dish"

def extract_tips(scraper):
    tips = set()
    tips.update([s for s in extract_sentences(scraper.description()) if is_actionable(s)])
    try:
        notes = scraper.notes()
        tips.update([s for s in extract_sentences(notes) if is_actionable(s)])
    except Exception:
        pass
    instructions = scraper.instructions()
    tips.update([s for s in extract_sentences(instructions) if is_actionable(s)])
    return list(tips)

output_file = "cooking_tips.jsonl"
sample_output = []

with open(output_file, "w", encoding="utf-8") as f:
    for url in sample_urls:
        try:
            scraper = scrape_me(url)
            title = scraper.title()
            tips = extract_tips(scraper)
            main_ingredient = extract_main_ingredient(title, tips)
            for tip in tips:
                # Use up to 3 prompt variations per tip
                used_templates = set()
                for template in prompt_templates:
                    if len(used_templates) >= 3:
                        break
                    prompt = template.format(title=title)
                    qa = {"prompt": prompt, "response": tip}
                    json.dump(qa, f, ensure_ascii=False)
                    f.write("\n")
                    used_templates.add(template)
                    if len(sample_output) < 10:
                        sample_output.append(qa)
                # Add 1 general prompt per tip
                for g_template in general_prompt_templates:
                    prompt = g_template.format(main_ingredient=main_ingredient)
                    qa = {"prompt": prompt, "response": tip}
                    json.dump(qa, f, ensure_ascii=False)
                    f.write("\n")
                    if len(sample_output) < 10:
                        sample_output.append(qa)
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

print("Sample Q&A pairs:")
for qa in sample_output:
    print(json.dumps(qa, ensure_ascii=False, indent=2))
print(f"\nExtraction complete. Tips saved to {output_file}.") 