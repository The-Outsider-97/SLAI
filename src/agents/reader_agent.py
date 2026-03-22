__version__ = "0.2.0"

import asyncio
from typing import Any, Dict, List

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.agents.base_agent import BaseAgent
from src.agents.reader import ParserEngine, ConversionEngine, RecoveryEngine, ReaderMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Reader Agent")
printer = PrettyPrinter


class ReaderAgent(BaseAgent):
    """Document workflow agent for parsing, conversion, recovery, and merge operations."""

    def __init__(self, shared_memory, agent_factory, config=None):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config,
        )
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.config = load_global_config()
        self.reader_agent_config = get_config_section("reader_agent")

        self.parser_engine = ParserEngine()
        self.conversion_engine = ConversionEngine()
        self.recovery_engine = RecoveryEngine(
            min_quality_score=self.reader_agent_config.get("recovery_min_quality_score", 0.55)
        )
        self.reader_memory = ReaderMemory()

        self.default_output_format = self.reader_agent_config.get("default_output_format", "txt")
        self.output_dir = self.reader_agent_config.get("output_dir", "output/reader")
        self.max_concurrency = max(1, self.reader_agent_config.get("max_concurrency", 4))
        self.enable_cache = self.reader_agent_config.get("enable_cache", True)

        logger.info(f"Reader Agent successfully Initialized")

    def _build_plan(self, instruction: str, files: List[str]) -> List[Dict[str, Any]]:
        lowered = (instruction or "").lower()
        plan: List[Dict[str, Any]] = [{"action": "parse", "files": files}]

        if any(word in lowered for word in ("recover", "repair", "corrupt", "broken")):
            plan.append({"action": "recover"})

        if "merge" in lowered or len(files) > 1:
            output = self._extract_format(instruction) or self.default_output_format
            plan.append({"action": "merge", "output": output})
        else:
            target = self._extract_format(instruction) or self.default_output_format
            plan.append({"action": "convert", "target": target})

        return plan

    def _extract_format(self, instruction: str) -> str | None:
        if not instruction:
            return None
        lowered = instruction.lower()
        supported = ["txt", "md", "html", "xml", "json", "csv", "pdf", "docx"]
        for fmt in supported:
            if f".{fmt}" in lowered or f" {fmt}" in lowered:
                return fmt
        return None

    async def _parse_files(self, files: List[str]) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def parse_one(file_path: str) -> Dict[str, Any]:
            cache_key = {"action": "parse", "file": file_path}
            if self.enable_cache:
                cached = self.reader_memory.get_cache(cache_key)
                if cached:
                    cached["cached"] = True
                    return cached

            async with semaphore:
                parsed = await asyncio.to_thread(self.parser_engine.parse, file_path)

            if self.enable_cache:
                self.reader_memory.set_cache(cache_key, parsed)
            parsed["cached"] = False
            return parsed

        return await asyncio.gather(*(parse_one(file_path) for file_path in files))

    async def _recover_docs(self, parsed_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def recover_one(doc: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                recovered = await asyncio.to_thread(self.recovery_engine.recover_document, doc)
            updated = dict(doc)
            updated["content"] = recovered["content"]
            updated["recovery"] = recovered
            return updated

        return await asyncio.gather(*(recover_one(doc) for doc in parsed_docs))

    async def _convert_docs(self, docs: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def convert_one(doc: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await asyncio.to_thread(
                    self.conversion_engine.convert,
                    doc,
                    target_format=target,
                    output_dir=self.output_dir,
                )

        return await asyncio.gather(*(convert_one(doc) for doc in docs))

    async def process(self, instruction: str, files: List[str]) -> Dict[str, Any]:
        plan = self._build_plan(instruction=instruction, files=files)
        self.reader_memory.write_checkpoint(
            "plan", {"instruction": instruction, "plan": plan, "files": files}
        )

        parsed_docs = await self._parse_files(files)

        for step in plan:
            action = step["action"]
            if action == "recover":
                parsed_docs = await self._recover_docs(parsed_docs)
            elif action == "convert":
                target = step.get("target", self.default_output_format)
                outputs = await self._convert_docs(parsed_docs, target)
                result = {"status": "ok", "plan": plan, "outputs": outputs}
                self.reader_memory.write_checkpoint("convert", result)
                return result
            elif action == "merge":
                output = step.get("output", self.default_output_format)
                merged = await asyncio.to_thread(
                    self.conversion_engine.merge,
                    parsed_docs,
                    output_format=output,
                    output_dir=self.output_dir,
                )
                result = {"status": "ok", "plan": plan, "merged": merged}
                self.reader_memory.write_checkpoint("merge", result)
                return result

        return {"status": "ok", "plan": plan, "outputs": parsed_docs}

    def perform_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        instruction = task_data.get("instruction", "")
        files = task_data.get("files", [])
        return asyncio.run(self.process(instruction=instruction, files=files))


if __name__ == "__main__":
    print("\n=== Running Reader Agent ===\n")
    printer.status("TEST", "Reader Agent initialized", "info")
    from src.agents.collaborative.shared_memory import SharedMemory
    from src.agents.agent_factory import AgentFactory
    shared_memory = SharedMemory()
    agent_factory = AgentFactory()

    agent = ReaderAgent(shared_memory=shared_memory,agent_factory=agent_factory)
    print(agent)

    print("\n* * * * * Phase 2 - Plan * * * * *\n")
    instruction = []
    file=None
    plan = agent._build_plan(instruction=instruction, files=file if file is not None else [])
    print(plan)

    print("\n=== Successfully ran the Reader Agent ===\n")