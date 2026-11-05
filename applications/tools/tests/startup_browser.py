"""Run against a freshly launched application's tokenized URL. Requires Playwright and Chromium."""
import argparse
import asyncio
import os

from playwright.async_api import async_playwright


async def check(url, name):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH"))
        try:
            for failed in (False, True):
                page = await browser.new_page(reduced_motion="reduce" if failed else "no-preference")
                entered, release = asyncio.Event(), asyncio.Event()

                async def gate(route):
                    entered.set()
                    await release.wait()
                    if failed:
                        await route.abort()
                    else:
                        await route.continue_()

                await page.route("**/api/**", gate)
                await page.goto(url, wait_until="commit")
                await asyncio.wait_for(entered.wait(), 30)
                splash = page.locator("dialog.slai-startup[open]")
                assert await splash.is_visible()
                assert await splash.get_by_role("status").text_content() == "Loading " + name
                await page.wait_for_function("document.querySelector('.slai-startup img')?.naturalWidth > 0")
                orb = splash.locator(".slai-startup-orb")
                motion = await orb.evaluate("(el) => getComputedStyle(el, '::before').animationName")
                assert motion == ("none" if failed else "slai-startup-orbit")
                release.set()
                if failed:
                    await splash.get_by_role("alert").wait_for()
                    await splash.get_by_role("button", name="Show application").click()
                await splash.wait_for(state="detached")
                assert await page.locator("#app, #root").first.inner_text()
                await page.close()
        finally:
            await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("name")
    args = parser.parse_args()
    asyncio.run(check(args.url, args.name))
