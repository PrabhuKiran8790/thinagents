from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import asyncio
import thinagents
from thinagents.core.response_models import ThinagentResponseStream
from typing import Any

agent: thinagents.Agent | None = None


def serialize_agent(agent: thinagents.Agent) -> dict:
    tools = getattr(agent, "tools", [])
    tool_names = [getattr(t, "name", str(t)) for t in tools]
    
    data = {
        "name": getattr(agent, "name", "Unnamed Agent"),
        "model": getattr(agent, "model", None),
        "tools": tool_names,
        "sub_agents": []
    }
    
    sub_agents = getattr(agent, "sub_agents", None) or getattr(agent, "children", None)
    if sub_agents:
        data["sub_agents"] = [serialize_agent(a) for a in sub_agents]
    
    return data


def format_chunk(chunk: ThinagentResponseStream[Any]) -> dict:
    """Format streaming chunk for SSE"""
    if hasattr(chunk, 'content_type'):
        content_type = chunk.content_type
        content = getattr(chunk, 'content', '')
        
        if content_type == 'tool_call':
            return {
                'type': 'tool_call',
                'tool_name': getattr(chunk, 'tool_name', 'unknown'),
                'content': content,
                'tool_call_args': getattr(chunk, 'tool_call_args', {})
            }
        elif content_type == 'tool_result':
            return {
                'type': 'tool_result',
                'tool_name': getattr(chunk, 'tool_name', 'unknown'),
                'content': content
            }
        elif content_type == 'completion':
            return {'type': 'text', 'content': ''}
        else:
            return {'type': 'text', 'content': content}
    
    if isinstance(chunk, str):
        return {'type': 'text', 'content': chunk}
    elif isinstance(chunk, dict):
        return {'type': 'text', 'content': chunk.get('content', '')}
    elif hasattr(chunk, 'content'):
        return {'type': 'text', 'content': chunk.content}
    else:
        return {'type': 'text', 'content': str(chunk)}


async def stream_agent(input_text):
    if agent is None:
        yield "Error: No agent configured"
        return
        
    if hasattr(agent, 'astream'):
        async for chunk in agent.astream(input_text, stream_intermediate_steps=True):
            yield chunk
    elif hasattr(agent, 'stream'):
        for chunk in agent.run(input_text, stream=True, stream_intermediate_steps=True):
            yield chunk
            await asyncio.sleep(0)
    elif hasattr(agent, 'arun'):
        result = await agent.arun(input_text, stream=True, stream_intermediate_steps=True)
        if hasattr(result, '__aiter__'):
            async for chunk in result:
                yield chunk
        else:
            yield result
    else:
        result = agent.run(input_text, stream=True, stream_intermediate_steps=True)
        if hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
            for chunk in result:
                yield chunk
                await asyncio.sleep(0)
        else:
            yield result


app = FastAPI(
    title="ThinAgents Web UI",
    description="Web interface for ThinAgents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/agent/info")
async def agent_info():
    if agent is None:
        return {"error": "No agent configured"}
    return serialize_agent(agent)


@app.post("/api/agent/run")
async def agent_run(request: Request):
    try:
        body = await request.json()
        input_text = body.get("input", "")
    except Exception:
        input_text = ""

    async def event_generator():
        try:
            async for chunk in stream_agent(input_text):
                data = format_chunk(chunk)
                if data:
                    yield f"data: {json.dumps(data)}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


ui_path = Path(__file__).parent.parent / "ui" / "build"

if ui_path.exists():
    app.mount("/_app", StaticFiles(directory=ui_path / "_app"), name="app")
    
    @app.get("/{full_path:path}")
    async def serve_ui(full_path: str = ""):
        if full_path.startswith("api/"):
            return
        
        file_path = ui_path / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        
        return FileResponse(ui_path / "index.html")
else:
    @app.get("/")
    async def root():
        return {
            "message": "ThinAgents Backend API (UI not built)",
            "agent": getattr(agent, "name", "Unnamed Agent") if agent else "No agent",
            "endpoints": {
                "info": "/api/agent/info",
                "run": "/api/agent/run"
            }
        }


def set_agent(agent_instance: thinagents.Agent):
    global agent
    agent = agent_instance


def create_app(agent_instance: thinagents.Agent):
    set_agent(agent_instance)
    return app