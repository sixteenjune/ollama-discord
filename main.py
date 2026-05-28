import asyncio
import contextlib
import json
import logging
import os
import re
import sqlite3
import time

import aiohttp
import discord

# bot settings and constants (optimised for 4gb vram)
DISCORD_TOKEN_ENV = "DISCORD_TOKEN"
OLLAMA_URL_ENV = "OLLAMA_URL"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "YOUR_MODEL"
REQUEST_TIMEOUT_SECONDS = 90
MAX_DISCORD_MESSAGE_LEN = 1900
RATE_LIMIT_SECONDS = 3
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.0
MEMORY_FILE_ENV = "MEMORY_FILE"
DEFAULT_MEMORY_FILE = "user_memory.sqlite"
MAX_MEMORY_MESSAGES = 16
MAX_FACT_KEYS = 20
MAX_HISTORY_MESSAGES = 10  
MAX_RESPONSE_SENTENCES = 2
MAX_RESPONSE_QUESTIONS = 1
BANNED_PHRASES = (
    "how can i help you",
    "what can i do for you today",
    "happy to help",
    "as an ai",
    "as a bot",
    "as a language model",
    "as an assistant",
    "i am an ai",
    "i'm an ai",
    "i am a bot",
    "i'm a bot",
    "i am an assistant",
    "i'm an assistant",
    "i can assist",
    "i can help",
    "how may i help",
)

CHARACTER_PROMPT = (
    # prompt goes here
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("discord_ollama_bot")


def _normalize_text(content: str) -> str:
    if not content:
        return ""
    return re.sub(
        r"^\s*(assistant|user|system)\s*:\s*",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()


# limits how fast users can talk to the bot to prevent spam
class RateLimiter:
    def __init__(self, cooldown_seconds: int) -> None:
        self._cooldown = cooldown_seconds
        self._last_used: dict[int, float] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = 0.0

    async def allow(self, user_id: int) -> bool:
        now = time.monotonic()
        async with self._lock:
            last = self._last_used.get(user_id, 0.0)
            if now - last < self._cooldown:
                return False
            self._last_used[user_id] = now
            if now - self._last_cleanup >= self._cooldown:
                stale_keys = [k for k, v in self._last_used.items() if now - v >= self._cooldown]
                for key in stale_keys:
                    self._last_used.pop(key, None)
                self._last_cleanup = now
            return True


# talks to the local ollama instance to get ai responses
class OllamaClient:
    def __init__(self, url: str, model: str, session: aiohttp.ClientSession) -> None:
        self._url = url
        self._model = model
        self._session = session

    async def generate(self, messages: list[dict[str, str]], stop_sequences: list[str]) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "stop": stop_sequences,
                "temperature": 0.6  # Lowered slightly for increased factual accuracy
            }
        }
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await asyncio.wait_for(
                    self._session.post(self._url, json=payload),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                async with response:
                    response.raise_for_status()
                    data = await response.json()
                    message = data.get("message") if isinstance(data, dict) else None
                    if not isinstance(message, dict) or "content" not in message:
                        raise ValueError("Invalid response format from Ollama /api/chat")
                    return str(message["content"])
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt >= MAX_RETRIES:
                    raise
                backoff = RETRY_BACKOFF_SECONDS * (2**attempt)
                logger.warning("Ollama request failed (%s). Retrying in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
        raise RuntimeError("Failed to get response from Ollama")


# manages short-term history and long-term user facts in a sqlite database
class MemoryManager:
    def __init__(self, file_path: str) -> None:
        self._db_path = self._normalize_db_path(file_path)
        self._db_lock = asyncio.Lock()
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()

    async def load_all_memory(self) -> None:
        await self._init_db()
        await self._migrate_legacy_history()

    async def clear_all_memory(self) -> None:
        """Safely wipes all records from the database."""
        async with self._db_lock:
            await asyncio.to_thread(self._clear_all_memory_sync)

    def _clear_all_memory_sync(self) -> None:
        self._conn.execute("DELETE FROM user_memory")
        self._conn.commit()
        self._conn.execute("VACUUM")
        self._conn.commit()

    # saves a new message to the user's short-term history
    async def append_message(self, user_id: int, role: str, content: str) -> None:
        content = _normalize_text(content)
        user_data = await self._get_user(str(user_id))
        user_data.setdefault("facts", {})
        history = user_data.setdefault("history", [])
        history.append({"role": role, "content": content})
        if len(history) > MAX_MEMORY_MESSAGES:
            user_data["history"] = history[-MAX_MEMORY_MESSAGES:]
        await self._set_user(str(user_id), user_data)

    # pulls out simple facts like the user's name or what they like
    async def extract_facts(self, user_id: int, content: str) -> None:
        name_value = self._extract_name(content)
        likes_value = self._extract_after_phrase(content, "i like")
        if not name_value and not likes_value:
            return
        user_data = await self._get_user(str(user_id))
        facts = user_data.setdefault("facts", {})
        if name_value:
            facts["name"] = name_value
        if likes_value:
            likes = [item.strip().lower() for item in likes_value.split(",") if item.strip()]
            if likes:
                existing = facts.get("likes", [])
                merged = list(dict.fromkeys(existing + likes))
                facts["likes"] = merged
        self._enforce_fact_limit(facts)
        await self._set_user(str(user_id), user_data)

    # glues the persona, exact user facts, and recent chat history all together for the ai
    async def build_prompt(self, user_id: int, username: str, user_message: str) -> list[dict[str, str]]:
        data = await self._get_user(str(user_id))
        facts = data.get("facts", {})
        history = data.get("history", [])
        facts_text = self._format_facts(facts)

        combined_instructions = (
            f"{CHARACTER_PROMPT}\n"
            f"---\n"
            f"current user information context:\n{facts_text}\n"
            f"---\n"
            f"the user talking to you is named: {username}"
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": combined_instructions}]

        for item in history[-MAX_HISTORY_MESSAGES:]:
            role = item.get("role", "user")
            content = item.get("content", "")
            if content.strip():
                messages.append({"role": role, "content": content.strip()})

        messages.append({"role": "user", "content": user_message.strip()})
        return messages

    async def _init_db(self) -> None:
        async with self._db_lock:
            await asyncio.to_thread(self._init_db_sync)

    async def _migrate_legacy_history(self) -> None:
        async with self._db_lock:
            await asyncio.to_thread(self._migrate_legacy_history_sync)

    def _migrate_legacy_history_sync(self) -> None:
        rows = self._conn.execute(
            "SELECT user_id, facts, history FROM user_memory"
        ).fetchall()
        if not rows:
            return
        updated = 0
        for user_id, facts_raw, history_raw in rows:
            history = self._safe_json_loads(history_raw, [])
            if not isinstance(history, list) or not history:
                continue
            cleaned = []
            changed = False
            for item in history:
                if not isinstance(item, dict):
                    cleaned.append(item)
                    continue
                role = item.get("role")
                content = item.get("content")
                if isinstance(content, str):
                    new_content = _normalize_text(content)
                    if new_content != content:
                        changed = True
                    cleaned.append({"role": role, "content": new_content})
                else:
                    cleaned.append(item)
            if changed:
                facts_json = facts_raw if facts_raw is not None else json.dumps({}, separators=(",", ":"))
                history_json = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
                self._conn.execute(
                    "INSERT INTO user_memory (user_id, facts, history) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(user_id) DO UPDATE SET facts = excluded.facts, history = excluded.history",
                    (user_id, facts_json, history_json),
                )
                updated += 1
        if updated:
            self._conn.commit()

    def _init_db_sync(self) -> None:
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS user_memory ("
            "user_id TEXT PRIMARY KEY,"
            "facts TEXT,"
            "history TEXT"
            ")"
        )
        self._conn.commit()

    def _format_facts(self, facts: dict) -> str:
        if not facts:
            return "(none)"
        lines = []
        for key, value in facts.items():
            if isinstance(value, list):
                lines.append(f"- {key}: {', '.join(value)}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    async def _get_user(self, user_id: str) -> dict:
        async with self._db_lock:
            return await asyncio.to_thread(self._get_user_sync, user_id)

    def _get_user_sync(self, user_id: str) -> dict:
        row = self._conn.execute(
            "SELECT facts, history FROM user_memory WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return {"facts": {}, "history": []}
        facts_raw, history_raw = row
        return {
            "facts": self._safe_json_loads(facts_raw, {}),
            "history": self._safe_json_loads(history_raw, []),
        }

    async def _set_user(self, user_id: str, data: dict) -> None:
        async with self._db_lock:
            await asyncio.to_thread(self._set_user_sync, user_id, data)

    def _set_user_sync(self, user_id: str, data: dict) -> None:
        facts_json = json.dumps(data.get("facts", {}), ensure_ascii=False, separators=(",", ":"))
        history_json = json.dumps(data.get("history", []), ensure_ascii=False, separators=(",", ":"))
        self._conn.execute(
            "INSERT INTO user_memory (user_id, facts, history) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET facts = excluded.facts, history = excluded.history",
            (user_id, facts_json, history_json),
        )
        self._conn.commit()

    def _enforce_fact_limit(self, facts: dict) -> None:
        if len(facts) <= MAX_FACT_KEYS:
            return
        keys = sorted(facts.keys())
        for key in keys:
            if len(facts) <= MAX_FACT_KEYS:
                break
            facts.pop(key, None)

    def _safe_json_loads(self, value: str | None, default):
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    def _normalize_db_path(self, file_path: str) -> str:
        if file_path.endswith(".sqlite"):
            return file_path
        return file_path + ".sqlite"

    def _extract_name(self, content: str) -> str:
        pattern = re.compile(
            r"\b(?:my name is|my name's|i am|i'm|call me)\s+([a-z]+(?:\s+[a-z]+){0,2})\b",
            flags=re.IGNORECASE,
        )
        match = pattern.search(content)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_after_phrase(self, content: str, phrase: str) -> str:
        pattern = re.compile(
            rf"\b{re.escape(phrase)}\b[\s:;,-]*[.!?]*\s*",
            flags=re.IGNORECASE,
        )
        match = pattern.search(content)
        if not match:
            return ""
        segment = content[match.end():]
        for delimiter in [".", "!", "?", "\n", ";"]:
            cut = segment.find(delimiter)
            if cut != -1:
                segment = segment[:cut]
                break
        return segment.strip()


# the main discord bot logic
class DiscordOllamaBot(discord.Client):
    def __init__(self, ollama_url: str, rate_limiter: RateLimiter, memory_manager: MemoryManager) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self._ollama_url = ollama_url
        self._ollama: OllamaClient | None = None
        self._session: aiohttp.ClientSession | None = None
        self._rate_limiter = rate_limiter
        self._memory = memory_manager
        self._fact_tasks: set[asyncio.Task] = set()

        self.tree = discord.app_commands.CommandTree(self)

        @self.tree.command(name="clear", description="Clear memory files")
        @discord.app_commands.default_permissions(administrator=True)
        async def clear_command(interaction: discord.Interaction):
            try:
                await self._memory.clear_all_memory()
                await interaction.response.send_message("memory cleared nya~", ephemeral=True)
            except Exception as e:
                logger.exception("Failed to clear memory")
                await interaction.response.send_message(f"error clearing memory: {e}", ephemeral=True)

    async def setup_hook(self) -> None:
        await self._memory.load_all_memory()
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
        )
        self._ollama = OllamaClient(self._ollama_url, OLLAMA_MODEL, self._session)
        await self.tree.sync()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        await super().close()

    async def on_ready(self) -> None:
        logger.info("Logged in as %s", self.user)

    # triggers every time a message is sent in a channel the bot can see
    async def on_message(self, message: discord.Message) -> None:
        if not self.user or message.author.id == self.user.id:
            return

        if self.user.id not in {mention.id for mention in message.mentions}:
            return

        prompt = self._extract_prompt(message.content)
        if not prompt:
            return

        self._start_fact_extraction(message.author.id, prompt)
        prompt = prompt[:MAX_DISCORD_MESSAGE_LEN]

        if not await self._rate_limiter.allow(message.author.id):
            logger.info("Rate limited user %s", message.author.id)
            await message.channel.send("wait a moment... nya~")
            return

        sender_username = message.author.display_name
        await self._memory.append_message(message.author.id, "user", prompt)
        
        messages_for_model = await self._memory.build_prompt(message.author.id, sender_username, prompt)
        logger.info("Request from %s compiled into prompt engine layout format", message.author.id)

        stop_sequences = [
            "<|im_end|>",
            "<|im_start|>",
            "assistant:",
        ]

        try:
            async with message.channel.typing():
                if not self._ollama:
                    raise RuntimeError("Ollama client not initialized")
                response_text = await self._ollama.generate(messages_for_model, stop_sequences)
        except asyncio.timeoutError:
            logger.exception("Ollama request timed out")
            await message.channel.send("*flicks tail anxiously* engine timed out... try again nya~")
            return
        except aiohttp.ClientError:
            logger.exception("Ollama request failed")
            await message.channel.send("connection failure meow...")
            return
        except Exception:
            logger.exception("Unexpected error calling Ollama")
            await message.channel.send("an error occurred...")
            return

        response_text = self._sanitize_response(response_text)

        if not response_text:
            await message.channel.send("...")
            return

        await self._memory.append_message(message.author.id, "assistant", response_text)
        
        for chunk in self._chunk_message(response_text):
            await message.channel.send(chunk)
        logger.info("Response sent to %s", message.author.id)

    def _start_fact_extraction(self, user_id: int, content: str) -> None:
        task = asyncio.create_task(self._memory.extract_facts(user_id, content))
        self._fact_tasks.add(task)
        task.add_done_callback(self._handle_fact_task)

    def _handle_fact_task(self, task: asyncio.Task) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            self._fact_tasks.discard(task)
            exc = task.exception()
            if exc:
                logger.exception("Fact extraction failed: %s", exc)

    def _extract_prompt(self, content: str) -> str:
        if not self.user:
            return ""
        mention_pattern = re.compile(rf"<@!?{self.user.id}>")
        cleaned = mention_pattern.sub("", content).strip()
        return cleaned

    def _chunk_message(self, text: str) -> list[str]:
        if len(text) <= MAX_DISCORD_MESSAGE_LEN:
            return [text]

        chunks: list[str] = []
        current = ""
        
        for line in text.splitlines(keepends=True):
            if len(current) + len(line) > MAX_DISCORD_MESSAGE_LEN:
                if current:
                    chunks.append(current.strip())
                current = line
            else:
                current += line
                
        if current:
            chunks.append(current.strip())
            
        return [c for c in chunks if c]

    def _sanitize_response(self, text: str) -> str:
        text = _normalize_text(text)
        if not text:
            return ""

        parts = re.split(r"(?<=[.!?])\s+", text)
        cleaned: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            part_l = part.lower()
            if any(phrase in part_l for phrase in BANNED_PHRASES):
                continue
            cleaned.append(part)
            if len(cleaned) >= MAX_RESPONSE_SENTENCES:
                break

        if not cleaned:
            if not parts:
                return ""
            first_part = parts[0].strip()
            if not first_part:
                return ""
            first_part_l = first_part.lower()
            if any(phrase in first_part_l for phrase in BANNED_PHRASES):
                return ""
            cleaned = [first_part]
        if not cleaned:
            return ""

        if sum("?" in part for part in cleaned) > MAX_RESPONSE_QUESTIONS:
            pruned: list[str] = []
            questions = 0
            for part in cleaned:
                if "?" in part:
                    questions += 1
                    if questions > MAX_RESPONSE_QUESTIONS:
                        continue
                pruned.append(part)
            cleaned = pruned if pruned else cleaned[:1]

        return " ".join(cleaned).strip()


def main() -> None:
    token = os.getenv(DISCORD_TOKEN_ENV)
    if not token:
        raise RuntimeError(f"Missing {DISCORD_TOKEN_ENV} environment variable")

    ollama_url = os.getenv(OLLAMA_URL_ENV, DEFAULT_OLLAMA_URL)
    memory_file = os.getenv(MEMORY_FILE_ENV, DEFAULT_MEMORY_FILE)
    rate_limiter = RateLimiter(RATE_LIMIT_SECONDS)
    memory_manager = MemoryManager(memory_file)
    client = DiscordOllamaBot(ollama_url, rate_limiter, memory_manager)
    client.run(token)


if __name__ == "__main__":
    main()
