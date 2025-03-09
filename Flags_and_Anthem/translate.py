import time
import json
import csv
import os
import re
import argparse
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# Create output directory
os.makedirs('anthem_translations', exist_ok=True)

# Read the CSV to get countries with anthem text
def read_countries_with_anthem(csv_file):
    countries = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['has_anthem_text'].lower() == 'true':
                    countries.append({
                        'code': row['country_code'],
                        'name': row['country_name']
                    })
        return countries
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

# Setup the webdriver
def setup_driver(headless=True):
    """Set up and return a Selenium webdriver"""
    options = Options()
    if headless:
        options.add_argument('--headless')
    
    # Add additional options to make the browser less detectable
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Use a realistic user agent
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
    ]
    options.add_argument(f'user-agent={random.choice(user_agents)}')
    
    # Install the webdriver manager to handle driver installation
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    # Add undetectable properties
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

# Function to clean and format verses
def clean_and_format_verses(verses, verbose=False):
    """Clean and format anthem verses for better readability"""
    if not verses:
        return []
        
    # Step 1: Strip out verse numbers if present
    cleaned_verses = []
    for verse in verses:
        # Skip very short verses and nav elements
        if len(verse) < 10 or "download" in verse.lower() or "back to top" in verse.lower():
            continue
            
        # Clean out verse numbers if present
        if re.match(r'^\d+\.', verse):
            # This verse starts with a number
            cleaned_verse = re.sub(r'^\d+\.\s*', '', verse)
            cleaned_verses.append(cleaned_verse)
        else:
            # Keep as is
            cleaned_verses.append(verse)
    
    # Step 2: Remove duplicates while preserving order
    seen = set()
    unique_verses = []
    for verse in cleaned_verses:
        if verse not in seen:
            seen.add(verse)
            unique_verses.append(verse)
    
    # Step 3: Check if verses are actually full verses or just lines
    # If they're all short (likely lines), group them into proper verses
    if all(len(v.split('\n')) <= 1 for v in unique_verses) and len(unique_verses) > 6:
        # These look like individual lines rather than verses
        # Try to group them into verses based on patterns
        verse_groups = []
        current_verse = []
        
        for line in unique_verses:
            if not line.strip():
                continue
                
            # Check if this looks like the start of a new verse
            if (not current_verse or 
                (len(line) < 40 and line[0].isupper()) or
                re.match(r'^[A-Z]', line)):
                # If we have a previous verse, add it
                if current_verse:
                    verse_groups.append('\n'.join(current_verse))
                # Start a new verse
                current_verse = [line]
            else:
                # Continue current verse
                current_verse.append(line)
        
        # Add the last verse
        if current_verse:
            verse_groups.append('\n'.join(current_verse))
        
        if verse_groups:
            unique_verses = verse_groups
    
    # Step 4: Try to identify and fix any obviously missing verse numbers
    if len(unique_verses) >= 3:
        # Check if the verses start with numbers
        has_numbers = [re.match(r'^\d+\.', v) for v in unique_verses]
        
        # If some have numbers and some don't, try to standardize
        if any(has_numbers) and not all(has_numbers):
            if verbose:
                print("Fixing inconsistent verse numbering...")
            
            fixed_verses = []
            for i, verse in enumerate(unique_verses, 1):
                if re.match(r'^\d+\.', verse):
                    # Already has a number, but may be wrong
                    fixed_verse = re.sub(r'^\d+\.', f"{i}.", verse)
                    fixed_verses.append(fixed_verse)
                else:
                    # Add a number
                    fixed_verses.append(f"{i}. {verse}")
            
            unique_verses = fixed_verses
    
    return unique_verses

# Function to extract text after clicking an element
def extract_text_after_click(driver, clickable_element, wait_time=2, verbose=False):
    """Extract text that appears after clicking an element"""
    # Save current page source for comparison
    before_source = driver.page_source
    
    try:
        # Scroll to the element and click it
        driver.execute_script("arguments[0].scrollIntoView(true);", clickable_element)
        time.sleep(0.5)
        clickable_element.click()
        time.sleep(wait_time)  # Wait for content to load
        
        # Get page source after clicking
        after_source = driver.page_source
        
        # Check if the page changed
        if after_source == before_source and verbose:
            print("Warning: Page source didn't change after clicking")
            
        # Look for newly visible elements (generally texts)
        visible_texts = []
        
        # Method 1: Look for common paragraph elements
        for tag in ['p', 'div', 'span']:
            elements = driver.find_elements(By.TAG_NAME, tag)
            for element in elements:
                try:
                    if element.is_displayed():
                        text = element.text.strip()
                        if len(text) > 20:  # Only substantive text
                            visible_texts.append(text)
                except:
                    continue
        
        # Method 2: Look for numbered paragraphs
        verse_pattern = re.compile(r'^\d+\.')
        for i in range(1, 10):
            xpath = f"//*[starts-with(text(), '{i}.')]"
            elements = driver.find_elements(By.XPATH, xpath)
            for element in elements:
                try:
                    if element.is_displayed():
                        text = element.text.strip()
                        if verse_pattern.match(text) and len(text) > 20:
                            visible_texts.append(text)
                except:
                    continue
        
        return visible_texts
    except Exception as e:
        if verbose:
            print(f"Error extracting text after click: {e}")
        return []

# Navigate to the homepage and find how to access country pages
def find_navigation_strategy(driver, base_url="https://nationalanthems.info/"):
    """Find how to navigate to country pages on the website"""
    try:
        print(f"Navigating to {base_url} to determine site structure...")
        driver.get(base_url)
        time.sleep(3)  # Give the page time to load
        
        # Check if there's a country list or menu
        country_links = driver.find_elements(By.XPATH, "//a[contains(@href, '.htm')]")
        if country_links:
            print(f"Found {len(country_links)} country links on homepage")
            
            # Check a few links to see if they follow country code pattern
            country_code_pattern = re.compile(r"/([a-z]{2})\.htm$")
            country_code_links = []
            
            for link in country_links[:20]:  # Check the first 20 links
                href = link.get_attribute('href')
                if href:
                    match = country_code_pattern.search(href)
                    if match:
                        country_code_links.append((match.group(1), href))
            
            if country_code_links:
                print(f"Found country code pattern in links: {country_code_links[:3]}")
                return "direct", None  # We can directly use country codes in URLs
        
        # Look for a search function
        search_input = driver.find_elements(By.XPATH, "//input[@type='search' or @type='text']")
        if search_input:
            print("Found search input on page")
            return "search", search_input[0]
            
        # Check if there's an alphabetical list or index
        alphabet_links = driver.find_elements(By.XPATH, "//a[contains(text(), 'A') or contains(text(), 'B') or contains(text(), 'C')]")
        if len(alphabet_links) >= 3:  # At least A, B, C
            print("Found alphabetical index")
            return "alphabetical", alphabet_links
        
        # Default to direct navigation and hope for the best
        print("Could not determine navigation strategy - will try direct URL")
        return "direct", None
        
    except Exception as e:
        print(f"Error determining navigation strategy: {e}")
        return "direct", None  # Default to direct navigation

# Function to scrape a country's anthem using Selenium
def scrape_anthem_translation(driver, country_code, country_name, base_url="https://nationalanthems.info/", 
                             nav_strategy="direct", nav_element=None, verbose=True):
    """Scrape anthem translation using Selenium"""
    try:
        if nav_strategy == "direct":
            # Try direct navigation with country code
            url = f"{base_url}{country_code}.htm"
            if verbose:
                print(f"Navigating directly to: {url}")
            driver.get(url)
            
        elif nav_strategy == "search" and nav_element:
            # Use search functionality
            if verbose:
                print(f"Searching for {country_name}")
            driver.get(base_url)
            wait = WebDriverWait(driver, 10)
            search_input = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='search' or @type='text']")))
            search_input.clear()
            search_input.send_keys(country_name)
            search_input.submit()
            
            # Wait for results and click on the country
            wait.until(EC.presence_of_element_located((By.XPATH, f"//a[contains(text(), '{country_name}')]")))
            country_link = driver.find_element(By.XPATH, f"//a[contains(text(), '{country_name}')]")
            country_link.click()
            
        elif nav_strategy == "alphabetical" and nav_element:
            # Navigate via alphabetical index
            first_letter = country_name[0].upper()
            if verbose:
                print(f"Navigating via alphabetical index: {first_letter}")
            driver.get(base_url)
            wait = WebDriverWait(driver, 10)
            letter_link = wait.until(EC.element_to_be_clickable((By.XPATH, f"//a[text()='{first_letter}']")))
            letter_link.click()
            
            # Find and click on the country
            wait.until(EC.presence_of_element_located((By.XPATH, f"//a[contains(text(), '{country_name}')]")))
            country_link = driver.find_element(By.XPATH, f"//a[contains(text(), '{country_name}')]")
            country_link.click()
        
        # Wait for page to load
        time.sleep(3)
        
        # Get current URL
        current_url = driver.current_url
        if verbose:
            print(f"Current URL: {current_url}")
        
        # Look for English translation section
        translation_found = False
        translation_text = ""
        verses = []
        english_elements = []  # Initialize this variable to avoid scope issues
        
        # Look for dropdown/accordion elements for English translation based on the screenshot
        translation_dropdown = None
        
        # Try to find the dropdown for English translation
        try:
            # Look specifically for the English translation dropdown/accordion
            dropdown_xpath_patterns = [
                "//a[contains(text(), 'English translation')]",  # Exact text as shown in screenshot
                "//span[contains(text(), 'English translation')]",
                "//div[contains(text(), 'English translation')]",
                "//*[contains(@class, 'collapsed') and contains(text(), 'English')]",  # Class might indicate collapsed state
                "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'english translation')]"
            ]
            
            for xpath in dropdown_xpath_patterns:
                elements = driver.find_elements(By.XPATH, xpath)
                if elements:
                    translation_dropdown = elements[0]
                    if verbose:
                        print(f"Found English translation dropdown: {translation_dropdown.text}")
                    break
            
            # If we found a dropdown, click it to expand
            if translation_dropdown:
                if verbose:
                    print("Clicking on dropdown to reveal English translation...")
                
                # Use our specialized function to get text after clicking
                extracted_texts = extract_text_after_click(driver, translation_dropdown, wait_time=2, verbose=verbose)
                
                if extracted_texts:
                    if verbose:
                        print(f"Extracted {len(extracted_texts)} text elements after clicking")
                    verses.extend(extracted_texts)
                else:
                    # If specialized function didn't work, use our previous approach
                    driver.execute_script("arguments[0].scrollIntoView(true);", translation_dropdown)
                    time.sleep(1)  # Give the page time to scroll
                    translation_dropdown.click()
                    time.sleep(2)  # Wait for expansion animation
                
                # After expanding, look for numbered verses as shown in screenshot
                verse_elements = []
                for i in range(1, 10):  # Look for verses 1-10
                    # Try multiple XPath patterns that match "1. Text", "1.Text", etc.
                    verse_xpath_patterns = [
                        f"//*[starts-with(normalize-space(text()), '{i}. ')]",  # "1. Text"
                        f"//*[starts-with(normalize-space(text()), '{i}.')]",   # "1.Text"
                        f"//p[starts-with(normalize-space(text()), '{i}. ')]",  # Paragraph starting with "1. "
                        f"//p[starts-with(normalize-space(text()), '{i}.')]"    # Paragraph starting with "1."
                    ]
                    
                    for xpath in verse_xpath_patterns:
                        elements = driver.find_elements(By.XPATH, xpath)
                        if elements:
                            for verse_el in elements:
                                try:
                                    if verse_el.is_displayed():  # Only get visible elements
                                        verse_text = verse_el.text.strip()
                                        if verse_text and len(verse_text) > 5:
                                            # Print the raw text for debugging
                                            if verbose:
                                                print(f"Raw verse {i} text: {verse_text[:50]}...")
                                            verses.append(verse_text)
                                            if verbose:
                                                print(f"Found verse {i}: {verse_text[:30]}...")
                                            break  # Found the verse, stop trying other XPaths
                                except Exception as e:
                                    if verbose:
                                        print(f"Error processing verse element: {e}")
                                        
                    # If we found at least one verse, that's a good sign
                    if len(verses) > 0 and i > 1 and len(verses) < i:
                        # We're missing verses. Try a different approach for remaining verses
                        if verbose:
                            print(f"Missing verses. Found {len(verses)} but looking for verse {i}")
                            
                # Special handling for anthem structure like in Algeria
                # If we found verses but they include the line numbers in the text,
                # do some cleanup to extract just the verse content
                cleaned_numbered_verses = []
                for i, verse in enumerate(verses):
                    # Check if this is a complete verse 
                    if re.match(r'^\d+\.', verse):
                        # This verse starts with a number
                        # Clean up the text - remove the number prefix
                        cleaned_verse = re.sub(r'^\d+\.\s*', '', verse)
                        cleaned_numbered_verses.append(cleaned_verse)
                    else:
                        # Keep as is
                        cleaned_numbered_verses.append(verse)
                
                # If we successfully cleaned the verses, use these instead
                if cleaned_numbered_verses:
                    verses = cleaned_numbered_verses
                
                # If we still don't have all verses, try direct extraction from page source
                if not verses or len(verses) < 3:  # Most anthems have at least 3 verses
                    if verbose:
                        print("Trying to extract verses from page source...")
                    try:
                        # Get the page source
                        page_source = driver.page_source
                        
                        # Look for verses in the form of "1. Verse text"
                        verse_pattern = r'(\d+\.\s*[^<>]{10,}?)(?=\d+\.|$)'
                        found_verses = re.findall(verse_pattern, page_source)
                        
                        if found_verses:
                            if verbose:
                                print(f"Found {len(found_verses)} verses in page source")
                            for verse in found_verses:
                                if verse not in verses:
                                    verses.append(verse.strip())
                    except Exception as e:
                        if verbose:
                            print(f"Error extracting from page source: {e}")
                
                # If no numbered verses found, look for paragraphs in the expanded section
                if not verses:
                    # Try to find the expanded container
                    expanded_container = None
                    
                    # Method 1: Look for parent with specific class that might indicate expanded state
                    try:
                        expanded_container = translation_dropdown.find_element(By.XPATH, "./following-sibling::div[1]")
                    except:
                        pass
                        
                    # Method 2: Look for container that appeared after clicking
                    if not expanded_container:
                        try:
                            # Get all text elements after the dropdown
                            text_elements = driver.find_elements(By.XPATH, "//p[string-length(text()) > 20]")
                            for el in text_elements:
                                if el.is_displayed():
                                    verse_text = el.text.strip()
                                    if verse_text and verse_text not in verses:
                                        verses.append(verse_text)
                        except:
                            pass
        except Exception as e:
            if verbose:
                print(f"Error handling dropdown: {e}")
        
        # If we didn't find verses through the dropdown, try regular methods
        if not verses:
            if verbose:
                print("Dropdown approach didn't yield verses, trying general methods...")
            
            # Look for English translation text elements (initialize the variable properly)
            english_elements = driver.find_elements(By.XPATH, 
                "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'english translation')]")
            
            if english_elements:
                if verbose:
                    print(f"Found {len(english_elements)} elements with 'English translation' text")
        
        if english_elements:
            if verbose:
                print(f"Found {len(english_elements)} elements with 'English translation' text")
            
            # Start with the first match
            translation_element = english_elements[0]
            
            # Try to find containing section
            parent_section = None
            current = translation_element
            for _ in range(5):  # Look up to 5 levels up
                if current.tag_name in ['div', 'section']:
                    parent_section = current
                    break
                current = current.find_element(By.XPATH, '..')
            
            # If we found a parent section, extract paragraphs from it
            if parent_section:
                if verbose:
                    print(f"Found parent section for translation")
                paragraphs = parent_section.find_elements(By.TAG_NAME, 'p')
                for p in paragraphs:
                    text = p.text.strip()
                    if text and 'English translation' not in text and len(text) > 5:
                        verses.append(text)
            else:
                # Otherwise try to find paragraphs that follow the translation element
                if verbose:
                    print(f"Looking for verses following the translation element")
                
                # Execute JavaScript to get all text nodes after the translation element
                script = """
                function getNextTextNodes(element, maxNodes = 20) {
                    const texts = [];
                    let current = element;
                    
                    while (current && texts.length < maxNodes) {
                        if (current.nodeType === Node.TEXT_NODE) {
                            const text = current.textContent.trim();
                            if (text.length > 5) {
                                texts.push(text);
                            }
                        } else if (current.nodeType === Node.ELEMENT_NODE) {
                            // Skip non-content elements
                            if (!['SCRIPT', 'STYLE', 'HEADER', 'NAV', 'FOOTER'].includes(current.tagName)) {
                                const text = current.textContent.trim();
                                if (text.length > 5) {
                                    texts.push(text);
                                }
                            }
                        }
                        current = current.nextSibling;
                    }
                    
                    return texts;
                }
                
                return getNextTextNodes(arguments[0]);
                """
                try:
                    next_texts = driver.execute_script(script, translation_element)
                    if next_texts:
                        verses.extend(next_texts)
                except Exception as js_error:
                    if verbose:
                        print(f"JavaScript error: {js_error}")
            
            # If we still don't have verses, look for verse numbers
            if not verses:
                if verbose:
                    print(f"Looking for numbered verses")
                verse_elements = driver.find_elements(By.XPATH, "//p[starts-with(text(), '1.')]")
                if verse_elements:
                    # We found at least the first verse, now look for more
                    for i in range(1, 10):  # Look for verses 1-9
                        verse_xpath = f"//p[starts-with(text(), '{i}.')]"
                        verse_elements = driver.find_elements(By.XPATH, verse_xpath)
                        if verse_elements:
                            verses.append(verse_elements[0].text.strip())
        
        # If no English translation section found, try looking for English text in various containers
        if not verses:
            if verbose:
                print(f"No specific 'English translation' section found, looking for English text")
            
            # Look for content in main containers
            for container in ['article', 'main', '.content', '#content', '.main', '#main']:
                try:
                    container_element = driver.find_element(By.CSS_SELECTOR, container)
                    paragraphs = container_element.find_elements(By.TAG_NAME, 'p')
                    for p in paragraphs:
                        text = p.text.strip()
                        if text and len(text) > 15:  # Longer text more likely to be verses
                            verses.append(text)
                    if verses:
                        break
                except NoSuchElementException:
                    continue
        
        # If we found verses, format them and return
        if verses:
            # Use our specialized cleaning function
            cleaned_verses = clean_and_format_verses(verses, verbose)
            
            # Check if we have any verses left after cleaning
            if cleaned_verses:
                # Join the verses with line breaks
                translation_text = "\n\n".join(cleaned_verses)
                
                if verbose:
                    print(f"Successfully extracted {len(cleaned_verses)} verses")
                    # Print a sample of the first verse for debugging
                    if cleaned_verses:
                        first_sample = cleaned_verses[0]
                        if len(first_sample) > 50:
                            first_sample = first_sample[:50] + "..."
                        print(f"Sample of first verse: {first_sample}")
                
                return {
                    'country_code': country_code,
                    'country_name': country_name,
                    'url': current_url,
                    'translation': translation_text,
                    'verses': len(cleaned_verses)
                }
        
        # If we made it here, we didn't find a valid translation
        if verbose:
            print(f"× No translation found for {country_name}")
        return None
        
        if verbose:
            print(f"× No translation found for {country_name}")
        return None
    
    except Exception as e:
        if verbose:
            print(f"× Error scraping {country_name}: {str(e)}")
        return None

# Save results to files
def save_results(results, verbose=True):
    """Save results to JSON, CSV, and individual text files"""
    # Save all results to a single JSON file
    json_path = 'anthem_translations/all_translations.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"Saved JSON data to {json_path}")
    
    # Create a CSV with the results
    csv_path = 'anthem_translations/anthems_english.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['country_code', 'country_name', 'url', 'verses', 'translation']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for data in results.values():
            writer.writerow(data)
    
    if verbose:
        print(f"Saved CSV data to {csv_path}")
    
    # Create a summary file
    summary_path = 'anthem_translations/summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"National Anthem Translations Summary\n")
        f.write(f"==================================\n\n")
        f.write(f"Total anthems scraped: {len(results)}\n\n")
        f.write(f"Countries with translations:\n")
        
        # Group by region/continent for better organization
        by_name = sorted(results.values(), key=lambda x: x['country_name'])
        for country in by_name:
            f.write(f"- {country['country_name']} ({country['country_code']}): {country['verses']} verses\n")
    
    if verbose:
        print(f"Saved summary to {summary_path}")

# Main function
def main(args):
    """Main function for the scraper"""
    # Get and display script parameters
    csv_file = args.file
    verbose = not args.quiet
    test_mode = args.test
    delay = args.delay
    headless = not args.visible
    
    if verbose:
        print(f"National Anthem Scraper (Selenium Version)")
        print(f"=========================================")
        print(f"CSV File: {csv_file}")
        print(f"Delay: {delay} seconds")
        print(f"Test Mode: {'Yes' if test_mode else 'No'}")
        print(f"Browser Mode: {'Visible' if not headless else 'Headless'}")
        print()
    
    # Get list of countries with anthem text
    countries = read_countries_with_anthem(csv_file)
    if not countries:
        print(f"No countries found in {csv_file} or file could not be read.")
        return
    
    # If in test mode, use only the first few countries
    if test_mode:
        test_count = min(3, len(countries))
        print(f"Test mode: Using only {test_count} of {len(countries)} countries")
        countries = countries[:test_count]
    else:
        print(f"Found {len(countries)} countries with anthem text to scrape")
    
    # Setup the browser
    try:
        driver = setup_driver(headless=headless)
        
        # Determine how to navigate the site
        base_url = "https://nationalanthems.info/"
        nav_strategy, nav_element = find_navigation_strategy(driver, base_url)
        
        # Results dictionary
        results = {}
        
        # Process each country
        for i, country in enumerate(countries, 1):
            print(f"[{i}/{len(countries)}] Scraping {country['name']}...")
            
            # Scrape the anthem translation
            result = scrape_anthem_translation(
                driver, 
                country['code'], 
                country['name'], 
                base_url, 
                nav_strategy, 
                nav_element, 
                verbose
            )
            
            if result:
                # Save to results dictionary
                results[country['code']] = result
                
                # Save to individual file
                with open(f"anthem_translations/{country['code']}_anthem.txt", 'w', encoding='utf-8') as f:
                    f.write(f"National Anthem of {country['name']}\n")
                    f.write(f"English Translation\n")
                    f.write("=" * 40 + "\n\n")
                    
                    # Format properly with verse numbers if needed
                    translation_text = result['translation']
                    # If the translation doesn't already have verse numbers but we detected verses
                    if result['verses'] > 1 and not re.search(r'^\d+\.', translation_text):
                        lines = translation_text.split('\n\n')
                        formatted_text = ""
                        for i, verse in enumerate(lines, 1):
                            if verse.strip():
                                formatted_text += f"{i}. {verse}\n\n"
                        f.write(formatted_text.strip())
                    else:
                        f.write(translation_text)
                    
                print(f"✓ Successfully saved translation for {country['name']} ({result['verses']} verses)")
            
            # Be respectful with rate limiting - pause between requests
            if i < len(countries):
                wait_time = delay + (i % 2)  # Slight randomization
                print(f"Waiting {wait_time} seconds before next request...")
                time.sleep(wait_time)
        
        # Save all results
        if results:
            save_results(results, verbose)
            print(f"\nCompleted! Scraped {len(results)} anthem translations.")
            print(f"Results saved to the anthem_translations directory.")
        else:
            print(f"\nNo translations were successfully scraped.")
    
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
    
    finally:
        # Ensure browser is closed
        try:
            driver.quit()
            print("Browser closed.")
        except:
            pass

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="National Anthem Scraper (Selenium Version)")
    parser.add_argument('--file', type=str, default='Flags_and_Anthem/data/national_data_collection_results.csv', help='CSV file with country data')
    parser.add_argument('--test', action='store_true', help='Run a test with just 3 countries')
    parser.add_argument('--delay', type=int, default=3, help='Delay between requests (seconds)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    parser.add_argument('--visible', action='store_true', help='Show browser window (not headless)')
    
    args = parser.parse_args()
    main(args)