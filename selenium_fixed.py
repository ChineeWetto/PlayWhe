#!/usr/bin/env python3
"""
Improved NLCB Play Whe Lottery Data Scraper using Selenium

This script uses Selenium to scrape historical Play Whe lottery data with improved error handling.
"""

import os
import time
import random
import logging
import re
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/selenium_fixed.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("fixed_play_whe_scraper")

class ImprovedPlayWheScraper:
    """
    An improved class to scrape Play Whe lottery results using Selenium
    """
    
    def __init__(self, base_url="https://www.nlcbplaywhelotto.com/nlcb-play-whe-results/", 
                 output_dir="data", 
                 delay_min=2, 
                 delay_max=5,
                 timeout=20):  # Increased timeout
        """
        Initialize the scraper with configuration parameters
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.timeout = timeout
        self.results = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Month and year mapping
        self.months = [
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ]
        
        # Initialize webdriver
        self.setup_driver()
        
    def setup_driver(self):
        """
        Set up the Selenium WebDriver with improved options
        """
        try:
            from selenium.webdriver.firefox.options import Options
            from selenium.webdriver.firefox.service import Service
            from webdriver_manager.firefox import GeckoDriverManager
            
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--width=1920')
            options.add_argument('--height=1080')
            options.set_preference("browser.download.folderList", 2)
            options.set_preference("browser.download.manager.showWhenStarting", False)
            options.set_preference("browser.download.dir", os.path.abspath(self.output_dir))
            
            # More stable settings
            options.set_preference("dom.max_script_run_time", 30)
            options.set_preference("dom.max_chrome_script_run_time", 30)
            
            service = Service(GeckoDriverManager().install())
            self.driver = webdriver.Firefox(service=service, options=options)
            self.driver.implicitly_wait(self.timeout)  # Using the timeout value
            logger.info("Firefox WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up WebDriver: {e}")
            raise
        
    def _random_delay(self):
        """
        Implement a random delay between requests
        """
        delay = random.uniform(self.delay_min, self.delay_max)
        logger.debug(f"Sleeping for {delay:.2f} seconds")
        time.sleep(delay)
        
    def navigate_to_base_url(self):
        """
        Navigate to the base URL with retry logic
        """
        max_retries = 3
        retries = 0
        
        while retries < max_retries:
            try:
                logger.info(f"Navigating to {self.base_url}")
                self.driver.get(self.base_url)
                
                # Wait for page to load with longer timeout
                WebDriverWait(self.driver, self.timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Check if the page loaded correctly
                if "Play Whe" in self.driver.page_source:
                    logger.info("Page loaded successfully")
                    return True
                else:
                    logger.warning("Page loaded but content doesn't appear to be correct")
                    retries += 1
                    
            except TimeoutException:
                logger.error(f"Timeout waiting for page to load (attempt {retries+1}/{max_retries})")
                retries += 1
                time.sleep(5)  # Wait before retry
            except Exception as e:
                logger.error(f"Error navigating to base URL: {e}")
                retries += 1
                time.sleep(5)  # Wait before retry
                
        logger.error(f"Failed to navigate to base URL after {max_retries} attempts")
        return False
            
    def search_month_year(self, month, year):
        """
        Search for Play Whe results for a specific month and year with improved robustness
        """
        logger.info(f"Searching for Play Whe results for {month} {year}")
        
        try:
            # Wait for the page to be fully loaded
            time.sleep(2)
            
            # Find the form elements with more robust selectors and explicit waits
            try:
                # Try to find the month select element with multiple approaches
                month_select = None
                selectors = [
                    "select[name='month']", 
                    "select:nth-of-type(1)", 
                    "form select:first-child"
                ]
                
                for selector in selectors:
                    try:
                        month_select = WebDriverWait(self.driver, 10).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        if month_select:
                            logger.info(f"Found month select with selector: {selector}")
                            break
                    except:
                        continue
                
                if not month_select:
                    raise NoSuchElementException("Could not find month select element with any selector")
                
                # Similar approach for year select
                year_select = None
                year_selectors = [
                    "select[name='year']",
                    "select:nth-of-type(2)",
                    "form select:nth-child(2)"
                ]
                
                for selector in year_selectors:
                    try:
                        year_select = WebDriverWait(self.driver, 10).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        if year_select:
                            logger.info(f"Found year select with selector: {selector}")
                            break
                    except:
                        continue
                
                if not year_select:
                    raise NoSuchElementException("Could not find year select element with any selector")
                
                # Select month and year with try-except blocks
                try:
                    Select(month_select).select_by_visible_text(month)
                except:
                    # Try partial text match if exact match fails
                    options = Select(month_select).options
                    for option in options:
                        if month in option.text:
                            option.click()
                            break
                
                try:
                    Select(year_select).select_by_value(year)
                except:
                    # Try by visible text if by value fails
                    try:
                        Select(year_select).select_by_visible_text(year)
                    except:
                        # Last resort - just click and use send_keys
                        year_select.click()
                        year_select.send_keys(year)
                
                # Find and click search button with multiple approaches
                search_button = None
                button_selectors = [
                    "input[type='submit'][value='SEARCH']",
                    "button[type='submit']",
                    "input[type='submit']",
                    "button:contains('Search')",
                    "form button"
                ]
                
                for selector in button_selectors:
                    try:
                        search_button = WebDriverWait(self.driver, 10).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        if search_button:
                            logger.info(f"Found search button with selector: {selector}")
                            break
                    except:
                        continue
                
                if not search_button:
                    # Last resort - try to submit the form directly
                    logger.warning("Could not find search button, trying to submit form directly")
                    form = self.driver.find_element(By.TAG_NAME, "form")
                    self.driver.execute_script("arguments[0].submit();", form)
                else:
                    # Click the search button
                    search_button.click()
                
            except Exception as e:
                logger.error(f"Error interacting with search form: {e}")
                # Try direct URL approach as fallback
                direct_url = f"{self.base_url}?month={month}&year={year}&search=SEARCH"
                logger.info(f"Trying direct URL approach: {direct_url}")
                self.driver.get(direct_url)
            
            # Wait for results to load with a longer timeout
            WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Add a longer delay to ensure results are fully loaded
            time.sleep(5)
            
            # Check if we got an error message or no results
            if "no results" in self.driver.page_source.lower() or "no records" in self.driver.page_source.lower():
                logger.warning(f"No results found for {month} {year}")
                return []
            
            # Parse the results with improved selector coverage
            month_results = self._parse_results()
            
            if month_results:
                logger.info(f"Extracted {len(month_results)} results for {month} {year}")
            else:
                logger.warning(f"No results could be parsed for {month} {year}")
                
            # Add delay before the next request
            self._random_delay()
            
            return month_results
            
        except Exception as e:
            logger.error(f"Error searching for {month} {year}: {e}")
            # Save screenshot for debugging
            try:
                screenshot_path = os.path.join(self.output_dir, f"error_{month}_{year}.png")
                self.driver.save_screenshot(screenshot_path)
                logger.info(f"Saved error screenshot to {screenshot_path}")
            except:
                pass
            return []
    
    def _parse_results(self):
        """
        Parse the Play Whe results with improved robustness
        """
        results = []
        page_source = self.driver.page_source
        
        try:
            # Try multiple selector approaches to find result containers
            selectors = [
                "div.play-whe-result", 
                "div.result-container",
                "div.results",
                "div.draw-results",
                "table.results-table",
                "div[class*='result']"
            ]
            
            # Try each selector until we find results
            result_sections = []
            for selector in selectors:
                try:
                    result_sections = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if result_sections:
                        logger.info(f"Found {len(result_sections)} result sections with selector: {selector}")
                        break
                except:
                    continue
            
            if not result_sections:
                logger.warning("No result sections found with any selector")
                
                # Last resort - try to parse based on common patterns in the HTML
                if re.search(r'\d{1,2}-[A-Za-z]{3}-\d{2}', page_source):
                    logger.info("Attempting to parse results from page source")
                    
                    # Extract dates
                    date_matches = re.findall(r'(\d{1,2})-([A-Za-z]{3})-(\d{2})', page_source)
                    
                    # Extract times and numbers using proximity in the source
                    for day, month_abbr, year_short in date_matches:
                        # Look for nearby content that might be results
                        date_pattern = f"{day}-{month_abbr}-{year_short}"
                        position = page_source.find(date_pattern)
                        
                        if position > 0:
                            # Extract a chunk of HTML after the date
                            chunk = page_source[position:position+500]
                            
                            # Look for time patterns
                            time_matches = re.findall(r'(\d{1,2}:\d{2}(AM|PM))', chunk)
                            
                            # Look for draw numbers
                            draw_matches = re.findall(r'Draw #(\d+)', chunk)
                            
                            # Look for winning numbers
                            num_matches = re.findall(r'Number[: ]*(\d{1,2})', chunk)
                            
                            if time_matches and draw_matches and num_matches:
                                for i in range(min(len(time_matches), len(draw_matches), len(num_matches))):
                                    try:
                                        full_year = f"20{year_short}" if int(year_short) >= 0 else f"19{year_short}"
                                        date_str = f"{full_year}-{self._month_to_number(month_abbr):02d}-{int(day):02d}"
                                        
                                        result = {
                                            'date': date_str,
                                            'time': time_matches[i][0],
                                            'number': int(num_matches[i]),
                                            'draw_number': int(draw_matches[i]),
                                            'day_of_week': datetime.strptime(date_str, '%Y-%m-%d').strftime('%A'),
                                            'month': month_abbr,
                                            'year': full_year
                                        }
                                        
                                        results.append(result)
                                    except:
                                        continue
                return results
            
            # Process each result section
            for section in result_sections:
                try:
                    # Extract date with multiple approaches
                    date_text = None
                    date_selectors = [
                        "h2.date-header", "div.date", "h2", "div.date-header",
                        "[class*='date']", "h2[class*='head']"
                    ]
                    
                    for selector in date_selectors:
                        try:
                            date_element = section.find_element(By.CSS_SELECTOR, selector)
                            date_text = date_element.text.strip()
                            if re.search(r'\d{1,2}-[A-Za-z]{3}-\d{2}', date_text):
                                break
                        except:
                            continue
                    
                    if not date_text or not re.search(r'\d{1,2}-[A-Za-z]{3}-\d{2}', date_text):
                        # Try getting any text from the section and parsing it
                        full_text = section.text
                        date_match = re.search(r'(\d{1,2})-([A-Za-z]{3})-(\d{2})', full_text)
                        if date_match:
                            date_text = date_match.group(0)
                        else:
                            continue
                    
                    date_match = re.search(r'(\d{1,2})-([A-Za-z]{3})-(\d{2})', date_text)
                    if not date_match:
                        continue
                        
                    day, month_abbr, year_short = date_match.groups()
                    full_year = f"20{year_short}" if int(year_short) >= 0 else f"19{year_short}"
                    date_str = f"{full_year}-{self._month_to_number(month_abbr):02d}-{int(day):02d}"
                    
                    # Find all draw sections with multiple approaches
                    draw_sections = []
                    draw_selectors = [
                        "div.draw-section", "div.draw", "div[class*='draw']",
                        "div.time-section", "div[class*='time']", "div.result-row"
                    ]
                    
                    for selector in draw_selectors:
                        try:
                            sections = section.find_elements(By.CSS_SELECTOR, selector)
                            if sections:
                                draw_sections = sections
                                break
                        except:
                            continue
                    
                    if not draw_sections:
                        # Try a generic approach - look for elements that might contain time and numbers
                        try:
                            # Find all divs that might be draw sections
                            divs = section.find_elements(By.TAG_NAME, "div")
                            for div in divs:
                                # Check if it contains time and number text
                                if (re.search(r'\d{1,2}:\d{2}', div.text) or 
                                    re.search(r'(AM|PM)', div.text)) and re.search(r'\d{1,2}', div.text):
                                    draw_sections.append(div)
                        except:
                            pass
                    
                    # Process each draw section
                    for draw_section in draw_sections:
                        try:
                            # Extract all the text from this section
                            section_text = draw_section.text
                            
                            # Extract time
                            time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM))', section_text)
                            if time_match:
                                time_text = time_match.group(1)
                            else:
                                # Try other common time formats
                                time_match = re.search(r'(\d{1,2}(?:AM|PM))', section_text)
                                if time_match:
                                    time_text = time_match.group(1)
                                else:
                                    time_match = re.search(r'(Morning|Midday|Afternoon|Evening)', section_text, re.IGNORECASE)
                                    if time_match:
                                        time_text = time_match.group(1)
                                    else:
                                        continue
                            
                            # Extract draw number
                            draw_match = re.search(r'Draw\s*#?\s*(\d+)', section_text, re.IGNORECASE)
                            if draw_match:
                                draw_number = int(draw_match.group(1))
                            else:
                                # Try finding any 4+ digit number (likely a draw number)
                                draw_match = re.search(r'\b(\d{4,})\b', section_text)
                                if draw_match:
                                    draw_number = int(draw_match.group(1))
                                else:
                                    continue
                            
                            # Extract winning number
                            # First try with specific formats
                            number_match = re.search(r'Number\s*:?\s*(\d{1,2})', section_text, re.IGNORECASE)
                            if not number_match:
                                # Try looking for standalone 1-2 digit numbers
                                numbers = re.findall(r'\b(\d{1,2})\b', section_text)
                                
                                # Filter to numbers 1-36
                                valid_numbers = [int(n) for n in numbers if 1 <= int(n) <= 36]
                                
                                if valid_numbers:
                                    # Use the first valid number that isn't the draw number
                                    for num in valid_numbers:
                                        if num != draw_number:
                                            number = num
                                            break
                                    else:
                                        continue
                                else:
                                    continue
                            else:
                                number = int(number_match.group(1))
                            
                            # Create result dictionary
                            result = {
                                'date': date_str,
                                'time': time_text,
                                'number': number,
                                'draw_number': draw_number,
                                'day_of_week': datetime.strptime(date_str, '%Y-%m-%d').strftime('%A'),
                                'month': month_abbr,
                                'year': full_year
                            }
                            
                            results.append(result)
                            
                        except Exception as e:
                            logger.warning(f"Error parsing draw section: {e}")
                            continue
                        
                except Exception as e:
                    logger.warning(f"Error parsing result section: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing results: {e}")
            
        return results
    
    def _month_to_number(self, month):
        """
        Convert month name to month number
        """
        try:
            return self.months.index(month) + 1
        except ValueError:
            # Try to match partial month name
            for i, m in enumerate(self.months):
                if month.lower() in m.lower():
                    return i + 1
            return 0
    
    def scrape_range(self, start_month, start_year, end_month=None, end_year=None):
        """
        Scrape Play Whe results for a range of months
        """
        logger.info(f"Starting Play Whe data scraping from {start_month} {start_year}")
        
        # Navigate to the base URL
        if not self.navigate_to_base_url():
            logger.error("Failed to navigate to base URL, aborting scrape")
            return pd.DataFrame()
        
        # If end month/year not provided, use current month/year
        if end_month is None or end_year is None:
            now = datetime.now()
            end_month = self.months[now.month - 1] if end_month is None else end_month
            end_year = str(now.year) if end_year is None else end_year
            
        # Convert month names to indices
        try:
            start_month_idx = self.months.index(start_month)
        except ValueError:
            logger.error(f"Invalid start month: {start_month}")
            return pd.DataFrame()
            
        try:
            start_year_idx = int(start_year)
        except ValueError:
            logger.error(f"Invalid start year: {start_year}")
            return pd.DataFrame()
            
        try:
            end_month_idx = self.months.index(end_month)
        except ValueError:
            logger.error(f"Invalid end month: {end_month}")
            return pd.DataFrame()
            
        try:
            end_year_idx = int(end_year)
        except ValueError:
            logger.error(f"Invalid end year: {end_year}")
            return pd.DataFrame()
        
        # Iterate through each month in the range
        current_year_idx = start_year_idx
        current_month_idx = start_month_idx
        
        while (current_year_idx < end_year_idx) or (current_year_idx == end_year_idx and current_month_idx <= end_month_idx):
            current_month = self.months[current_month_idx]
            current_year = str(current_year_idx)
            
            # Scrape the current month
            month_results = self.search_month_year(current_month, current_year)
            self.results.extend(month_results)
            
            # Save intermediate results after each month
            self._save_intermediate_results(f"improved_{current_month}_{current_year}")
            
            # Move to the next month
            current_month_idx += 1
            if current_month_idx >= len(self.months):
                current_month_idx = 0
                current_year_idx += 1
                
        # Close the driver
        try:
            self.driver.quit()
            logger.info("WebDriver closed successfully")
        except Exception as e:
            logger.error(f"Error closing WebDriver: {e}")
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        if not df.empty:
            # Sort by date, time, and draw number
            df = df.sort_values(by=['date', 'time', 'draw_number'])
            
            # Save to CSV
            csv_path = os.path.join(self.output_dir, 'improved_play_whe_results.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(df)} results to {csv_path}")
            
            # Save to JSON
            json_path = os.path.join(self.output_dir, 'improved_play_whe_results.json')
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Saved {len(df)} results to {json_path}")
        else:
            logger.warning("No results were scraped")
            
        return df
    
    def _save_intermediate_results(self, suffix):
        """
        Save intermediate results to avoid data loss
        """
        if not self.results:
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f'improved_play_whe_{suffix}.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved intermediate results ({len(df)} records) to {csv_path}")
        
if __name__ == "__main__":
    # Create scraper instance
    scraper = ImprovedPlayWheScraper()
    
    # Scrape more recent data first (more likely to work)
    # Change these dates to focus on more recent and reliable data
    results_df = scraper.scrape_range('Jan', '2022', 'Dec', '2024')
    
    # Print summary
    if not results_df.empty:
        print(f"Successfully scraped {len(results_df)} Play Whe results")
        print(f"Date range: {results_df['date'].min()} to {results_df['date'].max()}")
        print(f"Number of unique draw times: {results_df['time'].nunique()}")
        print(f"Number of unique numbers drawn: {results_df['number'].nunique()}")
        
        # Print frequency of each number
        number_freq = results_df['number'].value_counts().sort_index()
        print("\nNumber frequencies:")
        for number, count in number_freq.items():
            print(f"Number {number}: {count} occurrences")
    else:
        print("No results were scraped")