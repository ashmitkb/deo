"""
MCP bridge for Deo.

Connects to each server listed in config.MCP_SERVERS over stdio, pulls its
tool list, and converts each tool into the same {"type": "function", ...}
schema shape brain.py already sends to Ollama -- so from Ollama's point of
view there's no difference between a hand-written tool in tools.py and one
that lives on an MCP server. Brain routes the call back to the right place.

Setup:
    pip install mcp
    (individual MCP servers may need their own runtime, e.g. Node for
    @modelcontextprotocol/server-filesystem via npx)
"""

import asyncio
import contextlib
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from . import config


class MCPToolBridge:
    def __init__(self):
        self._sessions = {}      # server_name -> ClientSession
        self._namespace_of = {}  # tool_name -> server_name
        self._schemas = []
        self._loop = None
        self._ready = threading.Event()
        # Holds every stdio_client/ClientSession context open for the life of
        # the bridge, and guarantees they're all exited from the same task
        # that entered them -- entering with __aenter__() directly (as this
        # used to do) skips that guarantee and anyio throws a cancel-scope
        # error the moment anything tries to close.
        self._exit_stack = None

    def start(self):
        """Kick off connections to all configured MCP servers in a background
        thread/event loop and return immediately -- does NOT block startup.
        Tools simply aren't available (schemas() returns fewer/none) until
        each server finishes connecting, which is fine since Brain re-checks
        schemas() on every request rather than caching them once."""
        if not config.MCP_SERVERS:
            print("[mcp] no MCP_SERVERS configured -- skipping")
            self._ready.set()
            return

        threading.Thread(target=self._run_loop, daemon=True).start()

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_all())
        self._ready.set()
        self._loop.run_forever()

    async def _connect_all(self):
        self._exit_stack = contextlib.AsyncExitStack()
        for name, server_conf in config.MCP_SERVERS.items():
            try:
                params = StdioServerParameters(
                    command=server_conf["command"], args=server_conf.get("args", [])
                )
                read, write = await self._exit_stack.enter_async_context(stdio_client(params))
                session = await self._exit_stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                tool_list = await session.list_tools()

                self._sessions[name] = session
                for tool in tool_list.tools:
                    self._schemas.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema,
                        },
                    })
                    self._namespace_of[tool.name] = name
                print(f"[mcp] connected to '{name}' -- {len(tool_list.tools)} tool(s)")
            except Exception as e:
                print(f"[mcp] failed to connect to '{name}': {e}")

    def schemas(self) -> list:
        return self._schemas

    def has_tool(self, name: str) -> bool:
        return name in self._namespace_of

    def call_tool(self, name: str, args: dict) -> dict:
        server_name = self._namespace_of.get(name)
        session = self._sessions.get(server_name)
        if not session or not self._loop:
            return {"success": False, "message": f"No active MCP session for tool '{name}'"}
        try:
            future = asyncio.run_coroutine_threadsafe(
                session.call_tool(name, args), self._loop
            )
            result = future.result(timeout=20)
            text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
            return {"success": not result.isError, "message": text or "(no output)"}
        except Exception as e:
            return {"success": False, "message": f"MCP tool '{name}' raised an exception: {e}"}

    def stop(self):
        if not self._loop:
            return
        if self._exit_stack:
            fut = asyncio.run_coroutine_threadsafe(self._exit_stack.aclose(), self._loop)
            try:
                fut.result(timeout=5)
            except Exception as e:
                print(f"[mcp] error during shutdown: {e}")
        self._loop.call_soon_threadsafe(self._loop.stop)