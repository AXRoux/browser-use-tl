#!/usr/bin/env python3
"""
AI Model Comparison Agent

This script uses browser-use and Claude to perform
a comprehensive comparison between AI models.
"""

import os
import asyncio
from pathlib import Path
from langchain_anthropic import ChatAnthropic
from browser_use import Agent, Browser, BrowserConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if API key is loaded
if not os.getenv("ANTHROPIC_API_KEY"):
    print("ANTHROPIC_API_KEY not found in environment variables. Please ensure it's set in your .env file.")
    exit(1)

# Define the task
TASK = "Compare the price of gpt-4o and DeepSeek-V3"

async def main():
    # Create screenshot directory
    screenshot_dir = Path("./ai_model_screenshots")
    screenshot_dir.mkdir(exist_ok=True)
    print(f"Screenshots will be saved to {screenshot_dir.absolute()}")
    
    # Configure browser
    browser_config = BrowserConfig(
        headless=False,
    )
    
    # Create browser instance
    browser = Browser(config=browser_config)
    
    # Initialize the Anthropic model
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        temperature=0.7,
        max_tokens=4000,
        timeout=100
    )
    
    # Create the agent
    agent = Agent(
        task=TASK,
        llm=llm,
        browser=browser,
    )
    
    print("Starting AI model comparison agent...")
    
    # Run the agent
    result = await agent.run(max_steps=25)
    
    # Use correct methods to check completion status
    print(f"Agent task completed: {result.is_done()}")
    print(f"Agent encountered errors: {result.has_errors()}")
    
    # List URLs visited
    print("\nURLs visited:")
    for url in result.urls():
        print(f"- {url}")
    
    # List screenshots taken
    print("\nScreenshots taken:")
    for screenshot in result.screenshots():
        print(f"- {screenshot}")
    
    # Get final result if available
    if result.is_done() and not result.has_errors():
        print("\nFinal result:")
        print(result.final_result())
    
    # Close the browser
    await browser.close()

if __name__ == "__main__":
    asyncio.run(main()) 