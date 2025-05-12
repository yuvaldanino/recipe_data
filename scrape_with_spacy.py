import json
import re
import random
import spacy
from recipe_scrapers import scrape_me

nlp = spacy.load("en_core_web_sm")

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

# Prompt templates
prompt_templates = [
    "What's a tip for making {dish}?",
    "How do I make {dish} better?",
    "Any advice for cooking {dish}?",
    "How do I avoid common mistakes with {dish}?",
    "What should I know before making {dish}?",
    "How do I get the best results with {dish}?",
    "What's important to remember when making {dish}?",
    "What's a tip for cooking {main_ingredient}?",
    "How do I get the best results when cooking {main_ingredient}?"
]
# Dryness-related prompts (only use if tip mentions moisture/dryness)
dry_prompts = [
    "How do I avoid dry {main_ingredient}?"
]

dryness_keywords = ["dry", "moist", "moisture", "juicy", "overcooked", "undercooked", "tender", "succulent", "not dry", "not overcook", "not undercook", "keep moist", "keep juicy"]

# Actionable keywords and reason/explanation phrases
actionable_keywords = [
    "always", "never", "make sure", "let", "avoid", "important", "tip", "ensure", "best", "don't", "do not", "should", "must", "recommend", "careful", "keep", "use", "try", "allow", "rest", "cook", "bake", "stir", "whisk", "add", "remove", "check", "until", "before", "after", "because", "so that", "in order to", "to prevent", "to avoid", "to ensure", "why", "how"
]
reason_phrases = ["because", "so that", "in order to", "to prevent", "to avoid", "to ensure", "for best results", "so you can", "so you get", "so you have", "so you don't", "so you do not"]

# Helper: is the tip about dryness/moisture?
def is_dryness_tip(sentence):
    s = sentence.lower()
    return any(kw in s for kw in dryness_keywords)

# Helper: is the tip actionable and of sufficient quality?
def is_actionable(sentence):
    s = sentence.lower()
    # Must be at least 8 words or contain a reason/explanation phrase
    long_enough = len(s.split()) >= 8
    has_reason = any(phrase in s for phrase in reason_phrases)
    # Imperative or keyword match
    doc = nlp(sentence)
    imperative = len(doc) > 0 and doc[0].pos_ == "VERB"
    keyword_match = any(kw in s for kw in actionable_keywords)
    return (long_enough or has_reason) and (imperative or keyword_match)

def extract_sentences(text):
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def extract_main_ingredient(title, tips):
    food_words = ["chicken", "beef", "pasta", "egg", "eggs", "bread", "banana", "lasagna", "turkey", "spinach", "feta", "guacamole", "chili", "lentil", "soup", "enchilada", "steak", "rice", "cheese", "vegetable", "vegan", "meatball", "bolognese", "burger", "cake", "cookie", "salad", "fish", "shrimp", "pork", "lamb", "duck", "tofu", "tempeh", "bean", "pea", "potato", "carrot", "onion", "tomato", "pepper", "mushroom", "zucchini", "squash", "corn", "apple", "pear", "plum", "peach", "apricot", "berry", "strawberry", "blueberry", "raspberry", "blackberry", "cabbage", "broccoli", "cauliflower", "lettuce", "kale", "spinach", "arugula", "chard", "collard", "turnip", "parsnip", "radish", "celery", "leek", "garlic", "ginger", "herb", "basil", "cilantro", "parsley", "dill", "rosemary", "thyme", "sage", "oregano", "mint", "chive", "scallion", "shallot", "avocado", "guacamole", "chili", "lentil", "soup", "enchilada", "steak", "rice", "cheese", "vegetable", "vegan"]
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

def generalize_dish_name(title):
    title = re.sub(r"'s\s+", " ", title)
    title = re.sub(r"[A-Z][a-z]+\s+", "", title, count=1)
    title = re.sub(r"recipe", "", title, flags=re.IGNORECASE)
    title = re.sub(r"the best|best|easy|simple|classic|perfect|healthy|vegan|vegetarian|gluten[- ]?free|homemade|quick|ultimate|delicious|creamy|spicy|savory|sweet|tasty|authentic|traditional|easy|quick|one[- ]pot|one[- ]pan|oven[- ]baked|oven[- ]roasted|grilled|baked|fried|roasted|slow[- ]cooked|instant pot|air fryer|no[- ]bake|with.*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"[^a-zA-Z ]", "", title)
    title = title.strip().lower()
    return title if title else "the dish"

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

output_file = "cooking_tips_spacy.jsonl"
test_output_file = "cooking_tips_spacy_test.jsonl"
all_qa_pairs = []

for url in sample_urls:
    try:
        scraper = scrape_me(url)
        title = scraper.title()
        tips = extract_tips(scraper)
        main_ingredient = extract_main_ingredient(title, tips)
        dish = generalize_dish_name(title)
        for tip in tips:
            # Choose dryness prompt if tip is about dryness, else general prompt
            if is_dryness_tip(tip):
                template = random.choice(dry_prompts)
            else:
                template = random.choice(prompt_templates)
            prompt = template.format(dish=dish, main_ingredient=main_ingredient)
            qa = {"prompt": prompt, "response": tip}
            all_qa_pairs.append(qa)
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")

with open(output_file, "w", encoding="utf-8") as f:
    for qa in all_qa_pairs:
        json.dump(qa, f, ensure_ascii=False)
        f.write("\n")

sample_size = min(50, len(all_qa_pairs))
test_sample = random.sample(all_qa_pairs, sample_size)
with open(test_output_file, "w", encoding="utf-8") as f:
    for qa in test_sample:
        json.dump(qa, f, ensure_ascii=False)
        f.write("\n")

print(f"Extraction complete. {len(all_qa_pairs)} Q&A pairs saved to {output_file}.")
print(f"Test sample of {sample_size} Q&A pairs saved to {test_output_file}.") 