#!/usr/bin/env python3
"""
NLCB Play Whe Lottery Data Scraper using Selenium

This script uses Selenium to scrape historical Play Whe lottery data from nlcbplaywhelotto.com
by systematically querying month by month from July 1994 to present.

The script handles form interactions, error recovery, and rate limiting to ensure reliable data collection.
"""

import os
import time
import random
import logging
import re
import json
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
        logging.FileHandler("data/selenium_scraper.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("selenium_play_whe_scraper")

class SeleniumPlayWheScraper:
    """
    A class to scrape Play Whe lottery results from nlcbplaywhelotto.com using Selenium
    """
    
    def __init__(self, base_url="https://www.nlcbplaywhelotto.com/nlcb-play-whe-results/", 
                 output_dir="data", 
                 delay_min=2, 
                 delay_max=5):
        """
        Initialize the scraper with configuration parameters
        
        Args:
            base_url (str): The URL to scrape data from
            output_dir (str): Directory to save the scraped data
            delay_min (int): Minimum delay between requests in seconds
            delay_max (int): Maximum delay between requests in seconds
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.delay_min = delay_min
        self.delay_max = delay_max
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
        Set up the Selenium WebDriver using Firefox
        """
        from selenium.webdriver.firefox.options import Options
        from selenium.webdriver.firefox.service import Service
        from webdriver_manager.firefox import GeckoDriverManager
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--width=1920')
        options.add_argument('--height=1080')
        
        service = Service(GeckoDriverManager().install())
        self.driver = webdriver.Firefox(service=service, options=options)
        self.driver.implicitly_wait(10)
        
    def _random_delay(self):
        """
        Implement a random delay between requests to avoid overloading the server
        """
        delay = random.uniform(self.delay_min, self.delay_max)
        logger.debug(f"Sleeping for {delay:.2f} seconds")
        time.sleep(delay)
        
    def navigate_to_base_url(self):
        """
        Navigate to the base URL
        """
        logger.info(f"Navigating to {self.base_url}")
        self.driver.get(self.base_url)
        
        # Wait for page to load
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except TimeoutException:
            logger.error("Timeout waiting for page to load")
            
    def _parse_results(self):
        """
        Parse the Play Whe results from the current page
        
        Returns:
            list: List of dictionaries containing the parsed results
        """
        results = []
        
        try:
            # Find all result sections
            result_sections = self.driver.find_elements(By.CSS_SELECTOR, "div.play-whe-result, div.result-container")
            
            if not result_sections:
                logger.warning("No result sections found")
                return results
                
            # Process each result section
            for section in result_sections:
                try:
                    # Extract date
                    date_header = section.find_element(By.CSS_SELECTOR, "h2.date-header, div.date")
                    date_text = date_header.text.strip()
                    date_match = re.search(r'(\d{1,2})-([A-Za-z]{3})-(\d{2})', date_text)
                    
                    if not date_match:
                        continue
                        
                    day, month_abbr, year_short = date_match.groups()
                    full_year = f"20{year_short}" if int(year_short) >= 0 else f"19{year_short}"
                    date_str = f"{full_year}-{self._month_to_number(month_abbr):02d}-{int(day):02d}"
                    
                    # Find all draw sections
                    draw_sections = section.find_elements(By.CSS_SELECTOR, "div.draw-section, div.draw")
                    
                    for draw_section in draw_sections:
                        # Extract draw time
                        try:
                            time_header = draw_section.find_element(By.CSS_SELECTOR, "h3.time-header, div.time")
                            time_text = time_header.text.strip()
                        except NoSuchElementException:
                            continue
                            
                        # Extract draw number
                        try:
                            draw_info = draw_section.find_element(By.CSS_SELECTOR, "div.draw-info, div.draw-number")
                            draw_text = draw_info.text.strip()
                            draw_match = re.search(r'Draw #(\d+)', draw_text)
                            
                            if not draw_match:
                                continue
                                
                            draw_number = int(draw_match.group(1))
                        except NoSuchElementException:
                            continue
                            
                        # Extract winning number
                        try:
                            number_div = draw_section.find_element(By.CSS_SELECTOR, "div.number, span.winning-number")
                            number_text = number_div.text.strip()
                            try:
                                number = int(re.search(r'\d+', number_text).group())
                            except (ValueError, AttributeError):
                                continue
                        except NoSuchElementException:
                            continue
                            
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
                        
                except (NoSuchElementException, StaleElementReferenceException) as e:
                    logger.warning(f"Error parsing result section: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing results: {e}")
            
        return results
    
    def _month_to_number(self, month):
        """
        Convert month name to month number
        
        Args:
            month (str): Month name (e.g., 'Jan')
            
        Returns:
            int: Month number (1-12)
        """
        try:
            return self.months.index(month) + 1
        except ValueError:
            # Try to match partial month name
            for i, m in enumerate(self.months):
                if month.lower() in m.lower():
                    return i + 1
            return 0
    
    def search_month_year(self, month, year):
        """
        Search for Play Whe results for a specific month and year
        
        Args:
            month (str): Month name (e.g., 'Jan')
            year (str): Year (e.g., '2015')
            
        Returns:
            list: List of dictionaries containing the parsed results
        """
        logger.info(f"Searching for Play Whe results for {month} {year}")
        
        try:
            # Scroll to the search form
            search_form = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.search-form, form"))
            )
            self.driver.execute_script("arguments[0].scrollIntoView();", search_form)
            
            # Find month and year dropdowns
            month_select = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "select:nth-of-type(1)"))
            )
            year_select = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "select:nth-of-type(2)"))
            )
            
            # Select month and year
            Select(month_select).select_by_visible_text(month)
            Select(year_select).select_by_value(year)
            
            # Find and click search button
            search_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='submit'][value='SEARCH']"))
            )
            search_button.click()
            
            # Wait for results to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Add a small delay to ensure results are fully loaded
            time.sleep(2)
            
            # Parse the results
            month_results = self._parse_results()
            
            if month_results:
                logger.info(f"Extracted {len(month_results)} results for {month} {year}")
            else:
                logger.warning(f"No results found for {month} {year}")
                
            # Add delay before the next request
            self._random_delay()
            
            return month_results
            
        except (TimeoutException, NoSuchElementException, StaleElementReferenceException) as e:
            logger.error(f"Error searching for {month} {year}: {e}")
            return []
    
    def scrape_range(self, start_month, start_year, end_month=None, end_year=None):
        """
        Scrape Play Whe results for a range of months
        
        Args:
            start_month (str): Starting month name (e.g., 'Jan')
            start_year (str): Starting year (e.g., '2015')
            end_month (str, optional): Ending month name. If None, use current month.
            end_year (str, optional): Ending year. If None, use current year.
            
        Returns:
            pandas.DataFrame: DataFrame containing all scraped results
        """
        logger.info(f"Starting Play Whe data scraping from {start_month} {start_year}")
        
        # Navigate to the base URL
        self.navigate_to_base_url()
        
        # If end month/year not provided, use current month/year
        if end_month is None or end_year is None:
            now = datetime.now()
            end_month = self.months[now.month - 1] if end_month is None else end_month
            end_year = str(now.year) if end_year is None else end_year
            
        # Convert month names to indices
        start_month_idx = self.months.index(start_month)
        start_year_idx = int(start_year)
        end_month_idx = self.months.index(end_month)
        end_year_idx = int(end_year)
        
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
            self._save_intermediate_results(f"selenium_{current_month}_{current_year}")
            
            # Move to the next month
            current_month_idx += 1
            if current_month_idx >= len(self.months):
                current_month_idx = 0
                current_year_idx += 1
                
        # Close the driver
        self.driver.quit()
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        if not df.empty:
            # Sort by date, time, and draw number
            df = df.sort_values(by=['date', 'time', 'draw_number'])
            
            # Save to CSV
            csv_path = os.path.join(self.output_dir, 'selenium_play_whe_results.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(df)} results to {csv_path}")
            
            # Save to JSON
            json_path = os.path.join(self.output_dir, 'selenium_play_whe_results.json')
            df.to_json(json_path, orient='records', date_format='iso')
            logger.info(f"Saved {len(df)} results to {json_path}")
        else:
            logger.warning("No results were scraped")
            
        return df
    
    def _save_intermediate_results(self, suffix):
        """
        Save intermediate results to avoid data loss during long scraping sessions
        
        Args:
            suffix (str): Suffix to add to the filename
        """
        if not self.results:
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f'selenium_play_whe_results_{suffix}.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved intermediate results ({len(df)} records) to {csv_path}")
        
if __name__ == "__main__":
    # Create scraper instance
    scraper = SeleniumPlayWheScraper()
    
    # Scrape from July 1994 to present
    results_df = scraper.scrape_range('Jul', '1994')
    
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
