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

test_pass = 0
test_fail = 0

for url in sample_urls:
    print(f"\n--- Scraping: {url} ---")
    try:
        scraper = scrape_me(url)
        print("Title:", scraper.title())
        print("Description:", scraper.description())
        try:
            print("Notes:", scraper.notes())
        except Exception:
            print("Notes: Not available")
        print("Ingredients:", scraper.ingredients())
        print("Instructions:", scraper.instructions())
        print("Total Time:", scraper.total_time())
        print("Yields:", scraper.yields())
        test_pass += 1
    except Exception as e:
        print(f"Failed to scrape {url}: {e}") 
        test_fail += 1

print(f"\n--- Test Results ---")
print(f"Passed: {test_pass}")
print(f"Failed: {test_fail}")
