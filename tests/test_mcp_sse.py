"""Exercise the MCP HTTP transport without loading models or an information base."""

import asyncio
import importlib
import json
import sys
import types
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit


class AsgiExchange:
    def __init__(self, app, method, target, body=b""):
        self.app = app
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.messages = []
        self.incoming.put_nowait({"type": "http.request", "body": body})
        target_parts = urlsplit(target)
        self.scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1", "scheme": "http", "method": method,
            "path": target_parts.path, "raw_path": target_parts.path.encode(),
            "query_string": target_parts.query.encode(), "root_path": "",
            "headers": [(b"host", b"localhost:8000"), (b"content-type", b"application/json")],
            "server": ("localhost", 8000), "client": ("127.0.0.1", 12345),
        }

    async def send(self, message):
        if message["type"] == "http.response.start":
            if any(previous["type"] == "http.response.start" for previous in self.messages):
                raise AssertionError("ASGI response started twice")
        self.messages.append(message)
        await self.outgoing.put(message)

    async def run(self):
        await self.app(self.scope, self.incoming.get, self.send)

    async def next_event(self):
        while True:
            message = await asyncio.wait_for(self.outgoing.get(), 3)
            if message["type"] != "http.response.body":
                continue
            body = message.get("body", b"").decode()
            if body.startswith("event:"):
                fields = dict(line.split(": ", 1) for line in body.splitlines() if ": " in line)
                return fields


class McpSseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        app_context = types.ModuleType("code_search.app_context")
        app_context.IBManager = object
        app_context.IBConfig = object
        model_manager = types.ModuleType("code_search.model_manager")
        model_manager.ModelManager = object
        with patch.dict(sys.modules, {
            "code_search.app_context": app_context,
            "code_search.model_manager": model_manager,
        }):
            create_app = importlib.import_module("code_search.web.app").create_app
        self.app = create_app(object())

    async def post(self, target, payload):
        exchange = AsgiExchange(self.app, "POST", target, json.dumps(payload).encode())
        await asyncio.wait_for(exchange.run(), 3)
        return exchange

    async def test_initialize_list_tools_and_disconnect(self):
        from mcp.types import LATEST_PROTOCOL_VERSION

        # A second connection also works after the first client has disconnected.
        for connection_number in range(2):
            with self.subTest(connection=connection_number):
                stream = AsgiExchange(self.app, "GET", "/sse")
                stream_task = asyncio.create_task(stream.run())
                try:
                    endpoint_event = await stream.next_event()
                    self.assertEqual(endpoint_event["event"], "endpoint")
                    self.assertEqual(stream.messages[0]["status"], 200)
                    endpoint = endpoint_event["data"]
                    self.assertTrue(endpoint.startswith("/messages?session_id="))

                    accepted = await self.post(endpoint, {
                        "jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": LATEST_PROTOCOL_VERSION,
                                   "capabilities": {},
                                   "clientInfo": {"name": "transport-test", "version": "1"}},
                    })
                    self.assertEqual(accepted.messages[0]["status"], 202)
                    initialized = json.loads((await stream.next_event())["data"])
                    self.assertEqual(initialized["id"], 1)
                    self.assertEqual(initialized["result"]["serverInfo"]["name"], "1c-code-search")

                    await self.post(endpoint, {"jsonrpc": "2.0", "method": "notifications/initialized"})
                    await self.post(endpoint, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
                    tools_response = json.loads((await stream.next_event())["data"])
                    self.assertEqual(tools_response["id"], 2)
                    self.assertEqual(
                        {tool["name"] for tool in tools_response["result"]["tools"]},
                        {"search_code", "list_ibs"},
                    )
                finally:
                    await stream.incoming.put({"type": "http.disconnect"})
                    await asyncio.wait_for(stream_task, 3)

    async def test_message_without_session_has_one_error_response(self):
        response = await self.post("/messages", {})
        self.assertEqual(response.messages[0]["status"], 400)
        self.assertIn(b"session_id is required", response.messages[-1]["body"])


if __name__ == "__main__":
    unittest.main()
