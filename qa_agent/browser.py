from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urljoin, urlparse

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

logger = logging.getLogger("qa_agent.browser")

ViewportMode = Literal["desktop", "mobile"]
MOBILE_DEVICE_NAME = "iPhone 13"

INTERACTIVE_SELECTOR = (
    "a[href], button, input, textarea, select, "
    "[role='button'], [role='link'], [role='tab'], [role='menuitem'], "
    "[data-testid]"
)


@dataclass
class ElementRef:
    ref: str
    tag: str
    role: str
    name: str
    selector: str
    href: str = ""
    input_type: str = ""
    placeholder: str = ""
    visible: bool = True


@dataclass
class Observation:
    url: str
    title: str
    screenshot_b64: str
    dom_digest: str
    console_errors: list[str] = field(default_factory=list)
    network_errors: list[str] = field(default_factory=list)


class BrowserSession:
    """Thin Playwright wrapper for agent browser actions."""

    def __init__(
        self,
        base_url: str,
        *,
        headless: bool = True,
        viewport_mode: ViewportMode = "desktop",
        console_errors: list[str] | None = None,
        network_errors: list[str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headless = headless
        self.viewport_mode = viewport_mode
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._refs: dict[str, ElementRef] = {}
        self.console_errors = console_errors if console_errors is not None else []
        self.network_errors = network_errors if network_errors is not None else []

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser session is not started")
        return self._page

    async def start(self) -> None:
        self._playwright = await async_playwright().start()
        try:
            self._browser = await self._playwright.chromium.launch(headless=self.headless)
        except Exception as exc:
            logger.warning("Bundled Chromium unavailable (%s), falling back to system Chrome", exc)
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                channel="chrome",
            )
        if self.viewport_mode == "mobile":
            device = self._playwright.devices[MOBILE_DEVICE_NAME]
            self._context = await self._browser.new_context(
                **device,
                ignore_https_errors=True,
            )
        else:
            self._context = await self._browser.new_context(
                viewport={"width": 1440, "height": 900},
                ignore_https_errors=True,
            )
        self._page = await self._context.new_page()
        self._attach_listeners(self._page)

    async def close(self) -> None:
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

    def _attach_listeners(self, page: Page) -> None:
        def on_console(msg) -> None:
            if msg.type in ("error", "warning"):
                text = msg.text.strip()
                if text and text not in self.console_errors:
                    self.console_errors.append(text)

        def on_response(response) -> None:
            if response.status >= 400:
                entry = f"{response.status} {response.url}"
                if entry not in self.network_errors:
                    self.network_errors.append(entry)

        page.on("console", on_console)
        page.on("response", on_response)

    def _resolve_url(self, url: str) -> str:
        if url.startswith("http://") or url.startswith("https://"):
            return url
        if url.startswith("/"):
            return f"{self.base_url}{url}"
        return urljoin(f"{self.base_url}/", url)

    async def navigate(self, url: str) -> dict[str, Any]:
        target = self._resolve_url(url)
        response = await self.page.goto(target, wait_until="domcontentloaded", timeout=30_000)
        await self.page.wait_for_timeout(500)
        return {
            "url": self.page.url,
            "title": await self.page.title(),
            "status": response.status if response else None,
        }

    async def click(self, ref: str) -> dict[str, Any]:
        element = self._require_ref(ref)
        locator = self.page.locator(element.selector).first
        await locator.click(timeout=10_000)
        await self.page.wait_for_timeout(500)
        return {"clicked": ref, "name": element.name, "url": self.page.url}

    async def click_testid(self, testid: str) -> dict[str, Any]:
        locator = self.page.locator(f'[data-testid="{testid}"]').first
        await locator.click(timeout=10_000)
        await self.page.wait_for_timeout(500)
        return {"clicked_testid": testid, "url": self.page.url}

    async def select_option(self, ref: str, value: str) -> dict[str, Any]:
        element = self._require_ref(ref)
        if element.tag != "select":
            raise ValueError(f"Ref {ref} is not a select element (tag={element.tag})")
        locator = self.page.locator(element.selector).first
        await locator.select_option(value=value, timeout=10_000)
        await self.page.wait_for_timeout(300)
        return {"selected": ref, "value": value, "url": self.page.url}

    async def select_testid(self, testid: str, value: str) -> dict[str, Any]:
        locator = self.page.locator(f'select[data-testid="{testid}"]').first
        await locator.select_option(value=value, timeout=10_000)
        await self.page.wait_for_timeout(300)
        return {"selected_testid": testid, "value": value, "url": self.page.url}

    async def fill(self, ref: str, text: str) -> dict[str, Any]:
        element = self._require_ref(ref)
        locator = self.page.locator(element.selector).first
        await locator.fill(text, timeout=10_000)
        return {"filled": ref, "text_length": len(text)}

    async def press_key(self, key: str) -> dict[str, Any]:
        await self.page.keyboard.press(key)
        await self.page.wait_for_timeout(300)
        return {"pressed": key, "url": self.page.url}

    async def scroll(self, direction: str = "down", amount: int = 600) -> dict[str, Any]:
        delta = amount if direction == "down" else -amount
        await self.page.mouse.wheel(0, delta)
        await self.page.wait_for_timeout(300)
        return {"scrolled": direction, "amount": amount}

    async def go_back(self) -> dict[str, Any]:
        await self.page.go_back(wait_until="domcontentloaded", timeout=15_000)
        await self.page.wait_for_timeout(500)
        return {"url": self.page.url}

    async def wait_for(self, seconds: float = 2.0) -> dict[str, Any]:
        await self.page.wait_for_timeout(int(seconds * 1000))
        return {"waited_seconds": seconds, "url": self.page.url}

    async def set_appearance(
        self,
        *,
        theme_mode: str | None = None,
        color_scheme: str | None = None,
    ) -> dict[str, Any]:
        _ = (theme_mode, color_scheme)
        return {
            "success": False,
            "error": "set_appearance is not used in ml-vis playground",
            "url": self.page.url,
        }

    async def login(self, email: str, password: str) -> dict[str, Any]:
        _ = (email, password)
        return {
            "success": False,
            "error": "ml-vis playground has no login flow",
            "url": self.page.url,
        }

    async def clear_auth(self) -> None:
        if self._context:
            await self._context.clear_cookies()
            await self._context.clear_permissions()

    async def screenshot_png(self) -> bytes:
        return await self.page.screenshot(full_page=False, type="png")

    async def screenshot_b64(self) -> str:
        return base64.b64encode(await self.screenshot_png()).decode("ascii")

    async def build_dom_digest(self, max_elements: int | None = None) -> str:
        self._refs = {}
        if max_elements is None:
            max_elements = 60
        elements: list[ElementRef] = []
        locator = self.page.locator(INTERACTIVE_SELECTOR)
        count = await locator.count()
        for index in range(min(count, max_elements)):
            item = locator.nth(index)
            try:
                if not await item.is_visible():
                    continue
                tag = await item.evaluate("el => el.tagName?.toLowerCase?.() || ''")
                if tag in ("svg", "path", "g", "defs", "clippath", "symbol"):
                    continue
            except Exception:
                continue
            role = await item.get_attribute("role") or ""
            name = await self._element_name(item)
            href = await item.get_attribute("href") or ""
            input_type = await item.get_attribute("type") or ""
            placeholder = await item.get_attribute("placeholder") or ""
            selector = await self._build_selector(item, tag, index)

            ref_id = f"e{len(elements) + 1}"
            entry = ElementRef(
                ref=ref_id,
                tag=tag,
                role=role,
                name=name,
                selector=selector,
                href=href,
                input_type=input_type,
                placeholder=placeholder,
            )
            self._refs[ref_id] = entry
            elements.append(entry)

        page_context = await self._build_page_context()

        lines = [
            f"URL: {self.page.url}",
            f"Title: {await self.page.title()}",
        ]
        if page_context:
            lines.append("Page context:")
            lines.extend(f"  {line}" for line in page_context)
        lines.append("Interactive elements (use ref in click/fill/select_option):")
        for element in elements:
            parts = [element.ref, element.tag]
            if element.role:
                parts.append(f"role={element.role}")
            if element.name:
                parts.append(f'name="{element.name}"')
            if element.input_type:
                parts.append(f"type={element.input_type}")
            if element.placeholder:
                parts.append(f'placeholder="{element.placeholder}"')
            if element.href:
                parts.append(f"href={element.href}")
            if element.selector.startswith('[data-testid="'):
                parts.append(f"testid={element.selector.split('"')[1]}")
            lines.append(" | ".join(parts))

        return "\n".join(lines)

    async def _build_page_context(self) -> list[str]:
        return await self.page.evaluate(
            """() => {
                const lines = [];
                const canvas = document.querySelector('.playground-panel canvas');
                if (canvas) {
                    lines.push(`canvas: ${canvas.clientWidth}x${canvas.clientHeight}px`);
                }
                const epochPill = [...document.querySelectorAll('.metric-pill')]
                    .find((el) => /epoch/i.test(el.textContent || ''));
                if (epochPill) {
                    lines.push(`metrics: ${epochPill.textContent.trim()}`);
                }
                const pending = [...document.querySelectorAll('.metric-pill.pending')];
                if (pending.length) {
                    lines.push('config changed — reset or play before training');
                }
                const locale = document.querySelector('.locale-switcher select');
                if (locale) {
                    lines.push(`locale: ${locale.value}`);
                }
                const dataset = document.querySelector('.playground-controls select');
                if (dataset) {
                    lines.push(`dataset: ${dataset.value}`);
                }
                const hint = document.querySelector('.playground-hint');
                if (hint) {
                    lines.push(`hint: ${hint.textContent.trim()}`);
                }
                return lines;
            }"""
        )

    async def observe(self) -> Observation:
        return Observation(
            url=self.page.url,
            title=await self.page.title(),
            screenshot_b64=await self.screenshot_b64(),
            dom_digest=await self.build_dom_digest(),
            console_errors=list(self.console_errors[-10:]),
            network_errors=list(self.network_errors[-10:]),
        )

    def _require_ref(self, ref: str) -> ElementRef:
        element = self._refs.get(ref)
        if not element:
            raise ValueError(
                f"Unknown ref '{ref}'. Use refs from the latest DOM digest (e1, e2, ...)."
            )
        return element

    async def _element_name(self, locator) -> str:
        for attr in ("aria-label", "title", "alt", "name", "id"):
            try:
                value = await locator.get_attribute(attr)
            except Exception:
                value = None
            if value and value.strip():
                return value.strip()[:120]
        try:
            text = (await locator.inner_text()).strip()
        except Exception:
            text = ""
        if text:
            return " ".join(text.split())[:120]
        return ""

    async def _build_selector(self, locator, tag: str, index: int) -> str:
        element_id = await locator.get_attribute("id")
        if element_id:
            return f"#{element_id}"

        name = await locator.get_attribute("name")
        if name:
            return f'{tag}[name="{name}"]'

        href = await locator.get_attribute("href")
        if href and tag == "a":
            return f'a[href="{href}"]'

        data_testid = await locator.get_attribute("data-testid")
        if data_testid:
            return f'[data-testid="{data_testid}"]'

        return f"{INTERACTIVE_SELECTOR} >> nth={index}"


def format_tool_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)
