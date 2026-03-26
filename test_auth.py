import asyncio
from playwright.async_api import async_playwright
import os
import shutil

async def run():
    # Copy profile to avoid SQLite locks if the user's browser is open
    src = "/home/abby/.config/google-chrome"
    dst = "/home/abby/.config/google-chrome-bot"
    if os.path.exists(dst):
        shutil.rmtree(dst)
    try:
        shutil.copytree(src, dst, ignore_dangling_symlinks=True)
    except Exception as e:
        print(f"Warning on copy: {e}")

    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=dst,
            headless=True
        )
        page = await browser.new_page()
        print("Navigating to HN...")
        await page.goto("https://news.ycombinator.com/submit")
        await page.screenshot(path="hn_auth_test.png")
        print("Taking HN screenshot.")
        
        await page.goto("https://old.reddit.com/r/LocalLLaMA/submit")
        await page.screenshot(path="reddit_auth_test.png")
        print("Taking Reddit screenshot.")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
